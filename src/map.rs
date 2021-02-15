use mkit::thread::{Rx, Thread};

use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst},
        mpsc, Arc, RwLock,
    },
};

use crate::{
    entry::Entry,
    gc::{self, Cas, Epochs, Gc, GcThread, Reclaim},
    Result,
};

const SLOT_MASK: u64 = 0xFF;

pub struct Map<K, V> {
    id: usize,
    root: Arc<AtomicPtr<In<K, V>>>,
    epoch: Arc<AtomicU64>,
    access_log: Epochs,
    gc: Arc<Thread<Reclaim<K, V>>>,
    tx: Gc<K, V>,
}

pub struct In<K, V> {
    node: AtomicPtr<Node<K, V>>,
}

pub enum Node<K, V> {
    Trie {
        bmp: [u128; 2],
        childs: Vec<AtomicPtr<Child<K, V>>>,
    },
    List {
        head: AtomicPtr<Entry<K, V>>,
    },
}

pub enum Child<K, V> {
    Deep(In<K, V>),
    Leaf(Entry<K, V>),
}

impl<K, V> Map<K, V>
where
    K: 'static + Send,
    V: 'static + Send,
{
    pub fn new() -> Map<K, V> {
        let root = {
            let node = Box::new(Node::Trie {
                bmp: [0_u128; 2],
                childs: Vec::default(),
            });
            let inode = Box::new(In {
                node: AtomicPtr::new(Box::leak(node)),
            });
            Arc::new(AtomicPtr::new(Box::leak(inode)))
        };

        let access_log = Arc::new(RwLock::new(vec![Arc::new(AtomicU64::new(1))]));
        let log = Arc::clone(&access_log);
        let (gc, tx) = Thread::new("cmap", |rx: Rx<Reclaim<K, V>>| {
            let th = GcThread::new(log, rx);
            th
        });

        Map {
            id: 0,
            root,
            epoch: Arc::new(AtomicU64::new(1)),
            access_log,
            gc: Arc::new(gc),
            tx,
        }
    }
}

impl<K, V> Map<K, V> {
    pub fn cloned(&self) -> Map<K, V> {
        let id = {
            let mut access_log = self.access_log.write().expect("lock-panic");
            access_log.push(Arc::new(AtomicU64::new(1)));
            access_log.len().saturating_sub(1)
        };
        Map {
            id,
            root: Arc::clone(&self.root),
            epoch: Arc::clone(&self.epoch),
            access_log: Arc::clone(&self.access_log),
            gc: Arc::clone(&self.gc),
            tx: self.tx.clone(),
        }
    }

    fn generate_cas(&self) -> Cas<K, V> {
        let epoch = Arc::clone(&self.epoch);
        let at = {
            let access_log = self.access_log.read().expect("lock-panic");
            Arc::clone(&access_log[self.id])
        };
        Cas::new(epoch, at, &self.tx)
    }
}

impl<K, V> Node<K, V> {
    fn get_child(&self, n: usize) -> *mut Child<K, V> {
        match self {
            Node::Trie { childs, .. } => childs[n].load(SeqCst),
            Node::List { .. } => unreachable!(),
        }
    }

    fn hamming_distance(&self, w: u8) -> Option<Distance> {
        match self {
            Node::Trie { bmp, .. } => Some(hamming_distance(w, bmp.clone())),
            Node::List { .. } => None,
        }
    }

    fn new_list(k: &K, v: &V, leaf: &Entry<K, V>, cas: &mut Cas<K, V>) -> *mut Node<K, V>
    where
        K: Clone,
        V: Clone,
    {
        let tail_ptr = Box::leak(Box::new(Entry::N));
        let s2_ptr = Box::leak(Box::new(Entry::S {
            next: AtomicPtr::new(tail_ptr),
        }));

        let mut leaf = Box::new(leaf.cloned());
        leaf.update_next(s2_ptr);
        let leaf_ptr: *mut Entry<K, V> = Box::leak(leaf);

        let mut entry = Box::new(Entry::new(k.clone(), v.clone(), leaf_ptr));
        entry.update_next(leaf_ptr);
        let entry_ptr = Box::leak(entry);

        let s1_ptr = Box::leak(Box::new(Entry::S {
            next: AtomicPtr::new(entry_ptr),
        }));

        cas.free_on_fail(gc::Mem::Entry(tail_ptr));
        cas.free_on_fail(gc::Mem::Entry(s2_ptr));
        cas.free_on_fail(gc::Mem::Entry(leaf_ptr));
        cas.free_on_fail(gc::Mem::Entry(entry_ptr));
        cas.free_on_fail(gc::Mem::Entry(s1_ptr));

        let node_ptr = Box::leak(Box::new(Node::List {
            head: AtomicPtr::new(s1_ptr),
        }));
        cas.free_on_fail(gc::Mem::Node(node_ptr));

        node_ptr
    }

    fn new_subtrie(
        key: &K,
        value: &V,
        mut hash_pairs: Vec<(u8, u8)>,
        leaf: &Entry<K, V>,
        op: &mut CasOp<K, V>,
    ) -> *mut Node<K, V>
    where
        K: Clone,
        V: Clone,
    {
        if hash_pairs.len() == 0 {
            let node_ptr = Node::new_list(key, value, leaf, &mut op.cas);
            op.cas.free_on_fail(gc::Mem::Node(node_ptr));
            return node_ptr;
        }

        let (w1, w2) = hash_pairs.remove(0);

        if w1 == w2 {
            let mut bmp = [0_u128; 2];
            let off = if w1 < 128 { 0 } else { 1 };
            bmp[off] = 1 << w1;

            let node = Self::new_subtrie(key, value, hash_pairs, leaf, op);
            let child_ptr = Child::new_deep(node, &mut op.cas);
            let childs = vec![AtomicPtr::new(child_ptr)];

            let node_ptr = Box::leak(Box::new(Node::Trie { bmp, childs }));

            op.cas.free_on_fail(gc::Mem::Node(node_ptr));
            node_ptr
        } else {
            let mut bmp = [0_u128; 2];

            let off = if w1 < 128 { 0 } else { 1 };
            bmp[off] = 1 << w1;
            let child1_ptr = Child::new_leaf(key, value, &mut op.cas);

            let off = if w2 < 128 { 0 } else { 1 };
            bmp[off] = 1 << w2;
            let child2_ptr = Box::leak(Box::new(Child::Leaf(leaf.cloned())));

            let childs = vec![AtomicPtr::new(child1_ptr), AtomicPtr::new(child2_ptr)];
            let node_ptr = Box::leak(Box::new(Node::Trie { bmp, childs }));

            op.cas.free_on_fail(gc::Mem::Node(node_ptr));
            node_ptr
        }
    }

    fn update_list(key: &K, value: &V, mut op: CasOp<K, V>) -> CasRc<Option<V>>
    where
        K: PartialEq + Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::List { head } => {
                let head_ptr = head.load(SeqCst);
                let head_ref = unsafe { head_ptr.as_ref().unwrap() };

                let old_value = head_ref.set(&key, &value, &mut op.cas);

                let new = Box::leak(Box::new(Node::List {
                    head: AtomicPtr::new(head_ptr),
                }));

                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(old_value)
                } else {
                    CasRc::Retry
                }
            }
            Node::Trie { .. } => unreachable!(),
        }
    }

    fn ins_child(k: &K, v: &V, w: u8, n: usize, mut op: CasOp<K, V>) -> CasRc<()>
    where
        K: Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let bmp = {
                    let mut bmp = bmp.clone();
                    if w < 128 {
                        bmp[0] = bmp[0] | (1_u128 << w);
                    } else {
                        bmp[1] = bmp[1] | (1_u128 << w);
                    };
                    bmp
                };
                let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                    .iter()
                    .map(|c| AtomicPtr::new(c.load(SeqCst)))
                    .collect();

                let child_ptr = Child::new_leaf(k, v, &mut op.cas);
                childs.insert(n, AtomicPtr::new(child_ptr));
                let new = Box::leak(Box::new(Node::Trie { bmp, childs }));

                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
        }
    }

    fn set_leaf_child(k: &K, v: &V, n: usize, mut op: CasOp<K, V>) -> CasRc<()>
    where
        K: Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);

                let bmp = bmp.clone();
                let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                    .iter()
                    .map(|c| AtomicPtr::new(c.load(SeqCst)))
                    .collect();

                let new_child_ptr = Child::new_leaf(k, v, &mut op.cas);
                childs[n] = AtomicPtr::new(new_child_ptr);
                let new = Box::leak(Box::new(Node::Trie { bmp, childs }));

                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_fail(gc::Mem::Child(new_child_ptr));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
        }
    }

    fn set_trie_child(node: *mut Node<K, V>, n: usize, mut op: CasOp<K, V>) -> CasRc<()>
    where
        K: Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);

                let bmp = bmp.clone();
                let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                    .iter()
                    .map(|c| AtomicPtr::new(c.load(SeqCst)))
                    .collect();

                let new_child = Box::new(Child::Deep(In {
                    node: AtomicPtr::new(node),
                }));
                let new_child_ptr = Box::leak(new_child);
                childs[n] = AtomicPtr::new(new_child_ptr);
                let new = Box::leak(Box::new(Node::Trie { bmp, childs }));

                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_fail(gc::Mem::Child(new_child_ptr));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
        }
    }

    fn set_list(key: &K, value: &V, n: usize, mut op: CasOp<K, V>) -> CasRc<()>
    where
        K: Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));

                let bmp = bmp.clone();
                let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                    .iter()
                    .map(|c| AtomicPtr::new(c.load(SeqCst)))
                    .collect();

                let new = match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Leaf(leaf) => {
                        let node = Node::new_list(key, value, leaf, &mut op.cas);
                        childs[n] = AtomicPtr::new(Child::new_deep(node, &mut op.cas));
                        Box::leak(Box::new(Node::Trie { bmp, childs }))
                    }
                    Child::Deep(_) => unreachable!(),
                };
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
        }
    }
}

impl<K, V> Child<K, V> {
    fn new_leaf(key: &K, value: &V, cas: &mut gc::Cas<K, V>) -> *mut Child<K, V>
    where
        K: Clone,
        V: Clone,
    {
        let tail_ptr = Box::leak(Box::new(Entry::N));
        let child_ptr = {
            let entry = Entry::new(key.clone(), value.clone(), tail_ptr);
            Box::leak(Box::new(Child::Leaf(entry)))
        };
        cas.free_on_fail(gc::Mem::Entry(tail_ptr));
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }

    fn new_deep(node: *mut Node<K, V>, cas: &mut gc::Cas<K, V>) -> *mut Child<K, V> {
        let inode = In {
            node: AtomicPtr::new(node),
        };
        let child_ptr = Box::leak(Box::new(Child::Deep(inode)));
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }
}

impl<K, V> Map<K, V> {
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized + Hash,
    {
        {
            let access = self.epoch.load(SeqCst) | 0x8000000000000000;
            let access_log = self.access_log.read().expect("fail-lock");
            access_log[self.id].store(access, SeqCst)
        };

        let mut ws = key_to_hashbits(key);

        let mut inode: &In<K, V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

        let value = loop {
            let node = unsafe { inode.node.load(SeqCst).as_ref().unwrap() };
            inode = match (node, ws.pop()) {
                (Node::Trie { bmp, childs }, Some(w)) => {
                    let dist = hamming_distance(w, bmp.clone());
                    let child = match dist {
                        Distance::Set(n) => {
                            let child_ptr = childs[n].load(SeqCst);
                            unsafe { child_ptr.as_ref().unwrap() }
                        }
                        Distance::Insert(_) => break None,
                    };
                    match child {
                        Child::Deep(inode) => inode,
                        Child::Leaf(entry) if entry.borrow_key::<Q>().unwrap() == key => {
                            break Some(entry.as_value().clone());
                        }
                        Child::Leaf(_) => break None,
                    }
                }
                (Node::Trie { .. }, None) => unreachable!(),
                (Node::List { head }, _) => {
                    let head = unsafe { head.load(SeqCst).as_ref().unwrap() };
                    break Entry::get(head, key);
                }
            }
        };

        {
            let access_log = self.access_log.read().expect("fail-lock");
            access_log[self.id].store(self.epoch.load(SeqCst), SeqCst)
        };

        value
    }

    pub fn set<Q>(&self, key: K, value: V) -> Result<Option<V>>
    where
        K: PartialEq + Clone + Hash,
        V: Clone,
    {
        {
            let access = self.epoch.load(SeqCst) | 0x8000000000000000;
            let access_log = self.access_log.read().expect("fail-lock");
            access_log[self.id].store(access, SeqCst)
        };

        let mut ws = key_to_hashbits(&key);

        'retry: loop {
            let mut inode: &In<K, V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            loop {
                let old: *mut Node<K, V> = inode.node.load(SeqCst);
                let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };

                let w = match ws.pop() {
                    Some(w) => w,
                    None => {
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };
                        match Node::update_list(&key, &value, op) {
                            CasRc::Ok(old_value) => break 'retry Ok(old_value),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                };

                let n = match node.hamming_distance(w) {
                    Some(Distance::Insert(n)) => {
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };
                        match Node::ins_child(&key, &value, w, n, op) {
                            CasRc::Ok(_) => break 'retry Ok(None),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Some(Distance::Set(n)) => n,
                    None => unreachable!(),
                };

                let old_child_ptr = node.get_child(n);
                inode = match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Deep(inode) => inode,
                    Child::Leaf(leaf) if leaf.borrow_key().unwrap() == &key => {
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };
                        match Node::set_leaf_child(&key, &value, n, op) {
                            CasRc::Ok(_) => break 'retry Ok(None),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(_) if ws.len() == 0 => {
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };
                        match Node::set_list(&key, &value, n, op) {
                            CasRc::Ok(_) => break 'retry Ok(None),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(leaf) => {
                        let cas = self.generate_cas();
                        let mut op = CasOp { inode, old, cas };
                        let xs: Vec<(u8, u8)> = {
                            let leaf_key = leaf.borrow_key().unwrap();
                            let ls = key_to_hashbits(leaf_key)[..ws.len()].to_vec();
                            ws.clone().into_iter().zip(ls.into_iter()).collect()
                        };

                        let node_ptr = Node::new_subtrie(&key, &value, xs, leaf, &mut op);

                        match Node::set_trie_child(node_ptr, n, op) {
                            CasRc::Ok(_) => break 'retry Ok(None),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                }
            }
        }
    }
}

enum CasRc<T> {
    Ok(T),
    Retry,
}

struct CasOp<'a, K, V> {
    inode: &'a In<K, V>,
    old: *mut Node<K, V>,
    cas: Cas<'a, K, V>,
}

enum Distance {
    Set(usize),    // found
    Insert(usize), // not found
}

fn hamming_distance(w: u8, bmp: [u128; 2]) -> Distance {
    let posn = 1 << w;
    let mask = !(posn - 1);
    let bmp: u128 = if w < 128 { bmp[0] } else { bmp[1] };

    let (x, y) = ((bmp & mask), bmp);
    // TODO: optimize it with SSE or popcnt instructions, figure-out a way.
    let dist = (x ^ y).count_ones() as usize;

    match bmp & posn {
        0 => Distance::Insert(dist),
        _ => Distance::Set(dist),
    }
}

// TODO: Can we make this to use a generic hash function ?
#[inline]
fn key_to_hashbits<Q>(key: &Q) -> Vec<u8>
where
    Q: Hash + ?Sized,
{
    use fasthash::city;

    let mut hasher = city::Hasher64::default();
    key.hash(&mut hasher);
    let code: u64 = hasher.finish();

    (0..8)
        .map(|i| ((code >> (i * 8)) & SLOT_MASK) as u8)
        .collect()
}
