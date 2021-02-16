use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        mpsc, Arc, RwLock,
    },
    thread,
};

use crate::{
    gc::{self, gc_thread, Cas, Epochs, Reclaim},
    Result,
};

const SLOT_MASK: u64 = 0xFF;

// TODO: validate() method
//       * make sure that Node::List are only at the 8th level.
//       * make sure that there are no Node::Trie with empty childs.
// TODO: introspect() method

pub struct Map<K, V> {
    id: usize,
    root: Arc<AtomicPtr<In<K, V>>>,
    epoch: Arc<AtomicU64>,
    access_log: Epochs,
    handle: Arc<thread::JoinHandle<()>>,
    tx: mpsc::Sender<Reclaim<K, V>>,
}

pub struct In<K, V> {
    node: AtomicPtr<Node<K, V>>,
}

pub enum Node<K, V> {
    Trie {
        bmp: [u128; 2],
        childs: Vec<AtomicPtr<Child<K, V>>>,
    },
    Tomb {
        item: Item<K, V>,
    },
    List {
        items: Vec<Item<K, V>>,
    },
}

pub enum Child<K, V> {
    Deep(In<K, V>),
    Leaf(Item<K, V>),
}

#[derive(Clone)]
pub struct Item<K, V> {
    key: K,
    value: V,
}

impl<K, V> From<(K, V)> for Item<K, V> {
    fn from((key, value): (K, V)) -> Self {
        Item { key, value }
    }
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
        let (tx, rx) = mpsc::channel();

        let log = Arc::clone(&access_log);
        let handle = thread::spawn(move || gc_thread(log, rx));

        Map {
            id: 0,
            root,
            epoch: Arc::new(AtomicU64::new(1)),
            access_log,
            handle: Arc::new(handle),
            tx,
        }
    }

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
            handle: Arc::clone(&self.handle),
            tx: self.tx.clone(),
        }
    }
}

impl<K, V> Map<K, V> {
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
    fn new_list(items: Vec<Item<K, V>>, cas: &mut Cas<K, V>) -> *mut Node<K, V>
    where
        K: Clone,
        V: Clone,
    {
        let node_ptr = Box::leak(Box::new(Node::List { items }));
        cas.free_on_fail(gc::Mem::Node(node_ptr));
        node_ptr
    }

    fn new_subtrie(
        item: Item<K, V>,
        leaf: &Item<K, V>,
        mut pairs: Vec<(u8, u8)>,
        op: &mut CasOp<K, V>,
    ) -> *mut Node<K, V>
    where
        K: Clone,
        V: Clone,
    {
        if pairs.len() == 0 {
            return Node::new_list(vec![item, leaf.clone()], &mut op.cas);
        }

        let (w1, w2) = pairs.remove(0);

        if w1 == w2 {
            let mut bmp = [0_u128; 2];
            let off = if w1 < 128 { 0 } else { 1 };
            bmp[off] = 1 << w1;

            let node = Self::new_subtrie(item, leaf, pairs, op);
            let childs = vec![AtomicPtr::new(Child::new_deep(node, &mut op.cas))];

            let node_ptr = Box::leak(Box::new(Node::Trie { bmp, childs }));

            op.cas.free_on_fail(gc::Mem::Node(node_ptr));
            node_ptr
        } else {
            let mut bmp = [0_u128; 2];

            let off = if w1 < 128 { 0 } else { 1 };
            bmp[off] = 1 << w1;
            let child1_ptr = Box::leak(Box::new(Child::Leaf(item)));

            let off = if w2 < 128 { 0 } else { 1 };
            bmp[off] = 1 << w2;
            let child2_ptr = Box::leak(Box::new(Child::Leaf(leaf.clone())));

            let childs = vec![AtomicPtr::new(child1_ptr), AtomicPtr::new(child2_ptr)];
            let node_ptr = Box::leak(Box::new(Node::Trie { bmp, childs }));

            op.cas.free_on_fail(gc::Mem::Child(child1_ptr));
            op.cas.free_on_fail(gc::Mem::Child(child2_ptr));
            op.cas.free_on_fail(gc::Mem::Node(node_ptr));
            node_ptr
        }
    }

    fn to_tomb_item(&self) -> Option<Item<K, V>>
    where
        K: Clone,
        V: Clone,
    {
        match self {
            Node::Tomb { item } => Some(item.clone()),
            Node::Trie { .. } => None,
            Node::List { .. } => None,
        }
    }

    fn get_child(&self, n: usize) -> *mut Child<K, V> {
        match self {
            Node::Trie { childs, .. } => childs[n].load(SeqCst),
            Node::Tomb { .. } => unreachable!(),
            Node::List { .. } => unreachable!(),
        }
    }

    fn hamming_distance(&self, w: u8) -> Option<Distance> {
        match self {
            Node::Trie { bmp, .. } => Some(hamming_distance(w, bmp.clone())),
            Node::Tomb { .. } => None,
            Node::List { .. } => None,
        }
    }

    fn update_list(key: &K, value: &V, mut op: CasOp<K, V>) -> CasRc<Option<V>>
    where
        K: PartialEq + Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::List { items } => {
                let mut items = items.clone();
                let old_value = match items.iter_mut().find(|item| &item.key == key) {
                    Some(item) => {
                        let old_value = item.value.clone();
                        item.value = value.clone();
                        Some(old_value)
                    }
                    None => {
                        items.insert(0, (key.clone(), value.clone()).into());
                        None
                    }
                };

                let new = Box::leak(Box::new(Node::List { items }));

                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(old_value)
                } else {
                    CasRc::Retry
                }
            }
            Node::Trie { .. } => unreachable!(),
            Node::Tomb { .. } => unreachable!(),
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

                childs.insert(n, AtomicPtr::new(Child::new_leaf(k, v, &mut op.cas)));
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
            Node::Tomb { .. } => unreachable!(),
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

                childs[n] = AtomicPtr::new(Child::new_leaf(k, v, &mut op.cas));
                let new = Box::leak(Box::new(Node::Trie { bmp, childs }));

                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
            Node::Tomb { .. } => unreachable!(),
        }
    }

    fn set_list(k: &K, v: &V, n: usize, mut op: CasOp<K, V>) -> CasRc<()>
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

                let new = match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Leaf(leaf) => {
                        let items = vec![(k.clone(), v.clone()).into(), leaf.clone()];
                        let node = Node::new_list(items, &mut op.cas);
                        childs[n] = AtomicPtr::new(Child::new_deep(node, &mut op.cas));
                        Box::leak(Box::new(Node::Trie { bmp, childs }))
                    }
                    Child::Deep(_) => unreachable!(),
                };

                op.cas.free_on_pass(gc::Mem::Node(op.old));
                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));

                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
            Node::Tomb { .. } => unreachable!(),
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

                childs[n] = AtomicPtr::new(Child::new_deep(node, &mut op.cas));
                let new = Box::leak(Box::new(Node::Trie { bmp, childs }));

                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
            Node::Tomb { .. } => unreachable!(),
        }
    }
}

impl<K, V> Child<K, V> {
    fn new_leaf(key: &K, value: &V, cas: &mut gc::Cas<K, V>) -> *mut Child<K, V>
    where
        K: Clone,
        V: Clone,
    {
        let (key, value) = (key.clone(), value.clone());
        let child_ptr = Box::leak(Box::new(Child::Leaf((key, value).into())));
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
                    let child = match hamming_distance(w, bmp.clone()) {
                        Distance::Set(n) => {
                            let child_ptr = childs[n].load(SeqCst);
                            unsafe { child_ptr.as_ref().unwrap() }
                        }
                        Distance::Insert(_) => break None,
                    };
                    match child {
                        Child::Deep(inode) => inode,
                        Child::Leaf(item) if item.key.borrow() == key => {
                            break Some(item.value.clone());
                        }
                        Child::Leaf(_) => break None,
                    }
                }
                (Node::Trie { .. }, None) => unreachable!(),
                (Node::List { items }, _) => {
                    break items
                        .iter()
                        .find(|x| x.key.borrow() == key)
                        .map(|item| item.value.clone())
                }
                (Node::Tomb { item }, _) if item.key.borrow() == key => {
                    break Some(item.value.clone())
                }
                (Node::Tomb { .. }, _) => break None,
            }
        };

        {
            let access_log = self.access_log.read().expect("fail-lock");
            access_log[self.id].store(self.epoch.load(SeqCst), SeqCst)
        };

        value
    }

    pub fn set(&self, key: K, value: V) -> Result<Option<V>>
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
                    Child::Leaf(item) if item.key.borrow() == &key => {
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
                            let leaf_key = leaf.key.borrow();
                            let ls = key_to_hashbits(leaf_key)[..ws.len()].to_vec();
                            ws.clone().into_iter().zip(ls.into_iter()).collect()
                        };

                        let item: Item<K, V> = (key.clone(), value.clone()).into();
                        let node_ptr = Node::new_subtrie(item, leaf, xs, &mut op);

                        match Node::set_trie_child(node_ptr, n, op) {
                            CasRc::Ok(_) => break 'retry Ok(None),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                }
            }
        }
    }

    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Clone + Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized + Hash,
    {
        {
            let access = self.epoch.load(SeqCst) | 0x8000000000000000;
            let access_log = self.access_log.read().expect("fail-lock");
            access_log[self.id].store(access, SeqCst)
        };

        let ws = key_to_hashbits(&key);
        loop {
            let inode: &In<K, V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
            match self.do_remove(key, ws.clone(), inode) {
                RemRc::Some(old) => break Some(old),
                RemRc::None => break None,
                RemRc::Retry => (),
            }
        }
    }

    fn do_remove<Q>(&self, key: &Q, ws: Vec<u8>, inode: &In<K, V>) -> RemRc<V>
    where
        K: Clone + Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        'retry: loop {
            let old: *mut Node<K, V> = inode.node.load(SeqCst);
            let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };
            let mut slots = ws.clone();

            let w = match slots.pop() {
                Some(w) => w,
                None => {
                    let (new, ov) = match node {
                        Node::List { items } => {
                            let res = remove_from_list(key, items.as_slice());
                            match res {
                                Some((mut items, ov)) if items.len() == 1 => {
                                    let new = Node::Tomb {
                                        item: items.remove(0),
                                    };
                                    (new, ov)
                                }
                                Some((items, ov)) => (Node::List { items }, ov),
                                None => break 'retry RemRc::None,
                            }
                        }
                        Node::Tomb { .. } => break 'retry RemRc::Retry,
                        Node::Trie { .. } => unreachable!(),
                    };

                    let mut cas = self.generate_cas();

                    let new = Box::leak(Box::new(new));
                    cas.free_on_pass(gc::Mem::Node(old));
                    cas.free_on_fail(gc::Mem::Node(new));
                    if cas.swing(&inode.node, old, new) {
                        break 'retry RemRc::Some(ov);
                    } else {
                        continue 'retry;
                    }
                }
            };

            let (n, bmp, childs) = match node {
                Node::Trie { bmp, childs } => match hamming_distance(w, bmp.clone()) {
                    Distance::Insert(_) => return RemRc::None,
                    Distance::Set(n) => (n, bmp, childs),
                },
                Node::Tomb { .. } => break 'retry RemRc::Retry,
                Node::List { .. } => unreachable!(),
            };

            let old_child_ptr = childs[n].load(SeqCst);
            match unsafe { old_child_ptr.as_ref().unwrap() } {
                Child::Deep(inode) => match self.do_remove(key, slots, inode) {
                    RemRc::Retry => break 'retry RemRc::Retry,
                    RemRc::None => break 'retry RemRc::None,
                    RemRc::Some(old_value) => {
                        let mut cas = self.generate_cas();

                        let old: *mut Node<K, V> = inode.node.load(SeqCst);
                        let new = match node.to_tomb_item() {
                            Some(item) if childs.len() == 1 => Node::Tomb { item },
                            Some(item) => {
                                let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                                    .iter()
                                    .map(|c| AtomicPtr::new(c.load(SeqCst)))
                                    .collect();
                                let new_child_ptr = Box::leak(Box::new(Child::Leaf(item)));
                                childs[n] = AtomicPtr::new(new_child_ptr);

                                cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                                cas.free_on_fail(gc::Mem::Child(new_child_ptr));
                                Node::Trie {
                                    bmp: bmp.clone(),
                                    childs,
                                }
                            }
                            None => break 'retry RemRc::Some(old_value),
                        };
                        let new = Box::leak(Box::new(new));

                        cas.free_on_pass(gc::Mem::Node(old));
                        cas.free_on_fail(gc::Mem::Node(new));
                        if cas.swing(&inode.node, old, new) {
                            break 'retry RemRc::Some(old_value);
                        } else {
                            continue 'retry;
                        }
                    }
                },
                Child::Leaf(item) if item.key.borrow() == key => {
                    let mut cas = self.generate_cas();

                    let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();
                    childs.remove(n);

                    assert!(childs.len() > 0);
                    let new = match unsafe { childs[0].load(SeqCst).as_ref().unwrap() } {
                        Child::Leaf(item) if childs.len() == 1 => {
                            cas.free_on_pass(gc::Mem::Child(childs[0].load(SeqCst)));
                            Node::Tomb { item: item.clone() }
                        }
                        Child::Deep(_) | Child::Leaf(_) => Node::Trie {
                            bmp: bmp.clone(),
                            childs,
                        },
                    };
                    let new = Box::leak(Box::new(new));

                    cas.free_on_pass(gc::Mem::Node(old));
                    cas.free_on_fail(gc::Mem::Node(new));
                    if cas.swing(&inode.node, old, new) {
                        break 'retry RemRc::Some(item.value.clone());
                    } else {
                        continue 'retry;
                    }
                }
                Child::Leaf(_) => break 'retry RemRc::None,
            }
        }
    }
}

enum RemRc<V> {
    Some(V),
    None,
    Retry,
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

fn remove_from_list<K, V, Q>(key: &Q, items: &[Item<K, V>]) -> Option<(Vec<Item<K, V>>, V)>
where
    K: Clone + Borrow<Q>,
    V: Clone,
    Q: PartialEq + ?Sized,
{
    let res = items
        .iter()
        .enumerate()
        .find(|(_, x)| x.key.borrow() == key);

    match res {
        Some((i, x)) => {
            let mut xs = items[..i].to_vec();
            xs.extend_from_slice(&items[i + 1..]);
            Some((xs, x.value.clone()))
        }
        None => None,
    }
}
