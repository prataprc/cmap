use std::{
    borrow::Borrow,
    fmt::Debug,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        mpsc, Arc, RwLock,
    },
    thread,
};

use crate::{
    gc::{self, gc_thread, Cas, Epoch, Epochs, Reclaim},
    Result,
};

const SLOT_MASK: u64 = 0xFF;

// TODO: try_compact() may skip several tomb-nodes, if rw acess into Map is
//       focused on a specifi sub-tree. Implement a scavanger thread to clean-up
//       the tomb.
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

#[derive(Clone, PartialEq, Debug)]
pub struct Item<K, V> {
    key: K,
    value: V,
}

impl<K, V> From<(K, V)> for Item<K, V> {
    fn from((key, value): (K, V)) -> Self {
        Item { key, value }
    }
}

impl<K, V> Item<K, V>
where
    K: Debug,
    V: Debug,
{
    fn print(&self, prefix: &str) {
        println!("{}Item<{:?},{:?}>", prefix, self.key, self.value);
    }
}

impl<K, V> In<K, V>
where
    K: Debug,
    V: Debug,
{
    fn print(&self, prefix: &str) {
        let node = unsafe { self.node.load(SeqCst).as_ref().unwrap() };
        node.print(prefix)
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

        let epoch = Arc::new(AtomicU64::new(1));

        let access_log = Arc::new(RwLock::new(vec![Arc::new(AtomicU64::new(1))]));
        let (tx, rx) = mpsc::channel();

        let args = (Arc::clone(&epoch), Arc::clone(&access_log));
        let handle = thread::spawn(move || gc_thread(args.0, args.1, rx));

        Map {
            id: 0,
            root,
            epoch,
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

    pub fn print(&self)
    where
        K: Debug,
        V: Debug,
    {
        let epoch = self.epoch.load(SeqCst);
        let guard = self.access_log.write().expect("lock-panic");
        let access_log = guard.iter().map(|e| e.load(SeqCst));
        println!("Map<{},{:?}", epoch, access_log);
        unsafe { self.root.load(SeqCst).as_ref().unwrap().print("  ") };
    }
}

impl<K, V> Map<K, V> {
    fn generate_cas(&self) -> Cas<K, V> {
        Cas::new(&self.tx, Arc::clone(&self.epoch))
    }

    fn generate_epoch(&self) -> Epoch {
        let epoch = Arc::clone(&self.epoch);
        let at = {
            let access_log = self.access_log.read().expect("lock-panic");
            Arc::clone(&access_log[self.id])
        };
        Epoch::new(epoch, at)
    }
}

impl<K, V> Node<K, V> {
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

    fn hamming_set(&mut self, w: u8) {
        match self {
            Node::Trie { bmp, .. } => {
                let off = if w < 128 { 0 } else { 1 };
                bmp[off] = bmp[off] | 1 << (w % 128);
            }
            Node::Tomb { .. } | Node::List { .. } => (),
        }
    }

    fn hamming_reset(&mut self, w: u8) {
        match self {
            Node::Trie { bmp, .. } => {
                let off = if w < 128 { 0 } else { 1 };
                bmp[off] = bmp[off] & (!(1 << (w % 128)));
            }
            Node::Tomb { .. } | Node::List { .. } => (),
        }
    }

    fn print(&self, prefix: &str)
    where
        K: Debug,
        V: Debug,
    {
        match self {
            Node::Trie { bmp, childs } => {
                println!(
                    "{}Node::Trie<{:x}-{:x},{}>",
                    prefix,
                    bmp[1],
                    bmp[0],
                    childs.len()
                );
                let prefix = prefix.to_string() + "  ";
                for child in childs {
                    match unsafe { child.load(SeqCst).as_ref().unwrap() } {
                        Child::Leaf(item) => {
                            println!("{}Child::Leaf", prefix);
                            let prefix = prefix.to_string() + "  ";
                            item.print(&prefix);
                        }
                        Child::Deep(inode) => {
                            println!("{}Child::Inode", prefix);
                            let prefix = prefix.to_string() + "  ";
                            inode.print(&prefix);
                        }
                    }
                }
            }
            Node::Tomb { item } => {
                println!("{}Node::Tomb", prefix);
                let prefix = prefix.to_string() + "  ";
                item.print(&prefix);
            }
            Node::List { items } => {
                println!("{}Node::List", prefix);
                let prefix = prefix.to_string() + "  ";
                items.iter().for_each(|item| item.print(&prefix));
            }
        }
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
        key: &K,
        value: &V,
        leaf: &Item<K, V>,
        mut pairs: Vec<(u8, u8)>,
        op: &mut CasOp<K, V>,
    ) -> *mut Node<K, V>
    where
        K: Clone,
        V: Clone,
    {
        if pairs.len() == 0 {
            let item: Item<K, V> = (key.clone(), value.clone()).into();
            return Node::new_list(vec![item, leaf.clone()], &mut op.cas);
        }

        let (w1, w2) = pairs.pop().unwrap();
        println!("new subtrie:{:x},{:x}", w1, w2);

        if w1 == w2 {
            let mut node = {
                // one-child node trie, pointing to another intermediate node.
                let childs = {
                    let node = Self::new_subtrie(key, value, leaf, pairs, op);
                    vec![AtomicPtr::new(Child::new_deep(node, &mut op.cas))]
                };
                let bmp = [0_u128; 2];
                Box::new(Node::Trie { bmp, childs })
            };
            node.hamming_set(w1);
            let node_ptr = Box::leak(node);

            op.cas.free_on_fail(gc::Mem::Node(node_ptr));
            node_ptr
        } else {
            let child1 = AtomicPtr::new(Child::new_leaf_from(key, value, &mut op.cas));
            let (bmp, childs) = ([0_u128; 2], vec![child1]);
            let mut node = Node::Trie { bmp, childs };
            node.hamming_set(w1);

            let mut node = match node {
                Node::Trie { bmp, mut childs } => {
                    let n = match hamming_distance(w2, bmp.clone()) {
                        Distance::Insert(n) => n,
                        Distance::Set(_) => unreachable!(),
                    };
                    childs.insert(
                        n,
                        AtomicPtr::new(Child::new_leaf(leaf.clone(), &mut op.cas)),
                    );
                    Box::new(Node::Trie { bmp, childs })
                }
                _ => unreachable!(),
            };
            node.hamming_set(w2);
            let node_ptr = Box::leak(node);

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
            Node::List { items } => {
                let mut items = items.clone();
                let old_value = update_into_list(key, value, &mut items);

                let new = Box::leak(Box::new(Node::List { items }));

                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(old_value)
                } else {
                    CasRc::Retry
                }
            }
            Node::Tomb { .. } => CasRc::Retry,
            Node::Trie { .. } => unreachable!(),
        }
    }

    fn ins_child(key: &K, value: &V, w: u8, n: usize, mut op: CasOp<K, V>) -> CasRc<()>
    where
        K: Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let mut node = {
                    let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();
                    let new_child_ptr = Child::new_leaf_from(key, value, &mut op.cas);
                    childs.insert(n, AtomicPtr::new(new_child_ptr));

                    let bmp = bmp.clone();
                    Box::new(Node::Trie { bmp, childs })
                };
                node.hamming_set(w);

                let new = Box::leak(node);

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

    fn set_child(key: &K, value: &V, n: usize, mut op: CasOp<K, V>) -> CasRc<()>
    where
        K: Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                let node = {
                    let bmp = bmp.clone();
                    let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();
                    let new_child_ptr = Child::new_leaf_from(key, value, &mut op.cas);
                    childs[n] = AtomicPtr::new(new_child_ptr);
                    Box::new(Node::Trie { bmp, childs })
                };

                let new = Box::leak(node);

                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
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

    fn leaf_to_list(k: &K, v: &V, n: usize, mut op: CasOp<K, V>) -> CasRc<()>
    where
        K: Clone,
        V: Clone,
    {
        // convert a child node holding a leaf, into a interm-node pointing to node-list
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                let node = {
                    let bmp = bmp.clone();
                    let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();

                    let leaf = unsafe { old_child_ptr.as_ref().unwrap() }.to_leaf_item();
                    let new_child_ptr = {
                        let items = vec![(k.clone(), v.clone()).into(), leaf.clone()];
                        Child::new_deep(Node::new_list(items, &mut op.cas), &mut op.cas)
                    };

                    childs[n] = AtomicPtr::new(new_child_ptr);
                    Box::new(Node::Trie { bmp, childs })
                };

                let new = Box::leak(node);

                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
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

    fn set_trie_child(node: *mut Node<K, V>, n: usize, mut op: CasOp<K, V>) -> CasRc<()>
    where
        K: Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                let node = {
                    let bmp = bmp.clone();
                    let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();

                    childs[n] = AtomicPtr::new(Child::new_deep(node, &mut op.cas));
                    Box::new(Node::Trie { bmp, childs })
                };

                let new = Box::leak(node);

                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
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

    fn remove_from_list<Q>(key: &Q, mut op: CasOp<K, V>) -> CasRc<Option<V>>
    where
        K: Clone + Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        let (node, ov) = match unsafe { op.old.as_ref().unwrap() } {
            Node::List { items } => match remove_from_list(key, items) {
                Some((mut items, ov)) if items.len() == 1 => {
                    let item = items.remove(0); // remove the only item in the list
                    (Box::new(Node::Tomb { item }), ov)
                }
                Some((items, ov)) => (Box::new(Node::List { items }), ov),
                None => return CasRc::Ok(None),
            },
            Node::Tomb { .. } => return CasRc::Retry,
            Node::Trie { .. } => unreachable!(),
        };

        let new = Box::leak(node);

        op.cas.free_on_pass(gc::Mem::Node(op.old));
        op.cas.free_on_fail(gc::Mem::Node(new));
        if op.cas.swing(&op.inode.node, op.old, new) {
            CasRc::Ok(Some(ov))
        } else {
            CasRc::Retry
        }
    }

    fn remove_child(w: u8, n: usize, mut op: CasOp<K, V>) -> CasRc<Option<V>>
    where
        K: Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                let node = {
                    let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();
                    childs.remove(n);

                    let bmp = bmp.clone();
                    if childs.len() == 1 {
                        let item = {
                            let c = childs.remove(0).load(SeqCst);
                            unsafe { c.as_ref().unwrap() }.to_leaf_item()
                        };
                        Box::new(Node::Tomb { item })
                    } else {
                        let mut node = Box::new(Node::Trie { bmp, childs });
                        node.hamming_reset(w);
                        node
                    }
                };

                let new = Box::leak(node);

                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(&op.inode.node, op.old, new) {
                    CasRc::Ok(None)
                } else {
                    CasRc::Retry
                }
            }
            Node::Tomb { .. } => unreachable!(),
            Node::List { .. } => unreachable!(),
        }
    }

    fn try_compact(n: usize, item: Option<Item<K, V>>, mut op: CasOp<K, V>)
    where
        K: Clone,
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => match item {
                Some(item) => {
                    let old_child_ptr = childs[n].load(SeqCst);
                    let node = {
                        let bmp = bmp.clone();
                        let mut childs: Vec<AtomicPtr<Child<K, V>>> = childs
                            .iter()
                            .map(|c| AtomicPtr::new(c.load(SeqCst)))
                            .collect();
                        let new_child_ptr = Child::new_leaf(item, &mut op.cas);
                        childs[n] = AtomicPtr::new(new_child_ptr);

                        if childs.len() == 1 {
                            let item = {
                                let c = childs.remove(0).load(SeqCst);
                                unsafe { c.as_ref().unwrap() }.to_leaf_item()
                            };
                            Box::new(Node::Tomb { item })
                        } else {
                            Box::new(Node::Trie { bmp, childs })
                        }
                    };

                    let new = Box::leak(node);

                    op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                    op.cas.free_on_fail(gc::Mem::Node(new));
                    op.cas.free_on_pass(gc::Mem::Node(op.old));
                    op.cas.swing(&op.inode.node, op.old, new);
                }
                None => (),
            },
            Node::Tomb { .. } => unreachable!(),
            Node::List { .. } => unreachable!(),
        }
    }
}

impl<K, V> Child<K, V> {
    fn new_leaf_from(key: &K, value: &V, cas: &mut gc::Cas<K, V>) -> *mut Child<K, V>
    where
        K: Clone,
        V: Clone,
    {
        let (key, value) = (key.clone(), value.clone());
        let child_ptr = Box::leak(Box::new(Child::Leaf((key, value).into())));
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }

    fn new_leaf(leaf: Item<K, V>, cas: &mut gc::Cas<K, V>) -> *mut Child<K, V>
    where
        K: Clone,
        V: Clone,
    {
        let child_ptr = Box::leak(Box::new(Child::Leaf(leaf)));
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

    fn to_leaf_item(&self) -> Item<K, V>
    where
        K: Clone,
        V: Clone,
    {
        match self {
            Child::Leaf(leaf) => leaf.clone(),
            Child::Deep(_) => unreachable!(),
        }
    }
}

impl<K, V> Map<K, V> {
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Clone + Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized + Hash,
    {
        let _access_log = self.access_log.read().expect("fail-lock");
        let _epoch = self.generate_epoch();

        let mut ws = key_to_hashbits(key);
        let mut inode: &In<K, V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

        let value = loop {
            let old = inode.node.load(SeqCst);
            let node = unsafe { old.as_ref().unwrap() };

            let (n, child) = match (node, ws.pop()) {
                (Node::Trie { bmp, childs }, Some(w)) => {
                    let hd = hamming_distance(w, bmp.clone());
                    match hd {
                        Distance::Set(n) => {
                            let child_ptr = childs[n].load(SeqCst);
                            (n, unsafe { child_ptr.as_ref().unwrap() })
                        }
                        Distance::Insert(_) => break None,
                    }
                }
                (Node::Trie { .. }, None) => unreachable!(),
                (Node::List { items }, _) => break get_from_list(key, items),
                (Node::Tomb { item }, _) if item.key.borrow() == key => {
                    break Some(item.value.clone())
                }
                (Node::Tomb { .. }, _) => break None,
            };

            inode = match child {
                Child::Deep(next_inode) => {
                    let cas = self.generate_cas();
                    let mut op = CasOp { inode, old, cas };

                    let next_node = {
                        let next_node_ptr = next_inode.node.load(SeqCst);
                        op.cas.free_on_pass(gc::Mem::Node(next_node_ptr));
                        unsafe { next_node_ptr.as_ref().unwrap() }
                    };
                    match next_node.to_tomb_item() {
                        Some(item) if item.key.borrow() == key => {
                            let value = item.value.clone();
                            Node::try_compact(n, Some(item), op);
                            break Some(value);
                        }
                        Some(item) => {
                            Node::try_compact(n, Some(item), op);
                            break None;
                        }
                        None => next_inode,
                    }
                }
                Child::Leaf(item) if item.key.borrow() == key => {
                    break Some(item.value.clone());
                }
                Child::Leaf(_) => break None,
            };
        };

        value
    }

    pub fn set(&self, key: K, value: V) -> Result<Option<V>>
    where
        K: PartialEq + Clone + Hash,
        V: Clone,
    {
        let _access_log = self.access_log.read().expect("fail-lock");
        let _epoch = self.generate_epoch();

        let mut ws = key_to_hashbits(&key);

        println!(
            "set outer {:?}",
            ws.iter()
                .map(|w| format!("{:x}", w))
                .collect::<Vec<String>>()
        );

        'retry: loop {
            let mut inode: &In<K, V> = unsafe {
                //
                self.root.load(SeqCst).as_ref().unwrap()
            };

            loop {
                let old: *mut Node<K, V> = inode.node.load(SeqCst);
                let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };

                let w = match ws.pop() {
                    Some(w) => w,
                    None => {
                        // must be Node::List
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };

                        match Node::update_list(&key, &value, op) {
                            CasRc::Ok(old_value) => break 'retry Ok(old_value),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                };
                println!("set loop w:{:x}", w);

                let n = match node {
                    Node::Trie { bmp, .. } => match hamming_distance(w, bmp.clone()) {
                        Distance::Insert(n) => {
                            let cas = self.generate_cas();
                            let op = CasOp { inode, old, cas };

                            match Node::ins_child(&key, &value, w, n, op) {
                                CasRc::Ok(_) => break 'retry Ok(None),
                                CasRc::Retry => continue 'retry,
                            }
                        }
                        Distance::Set(n) => n,
                    },
                    Node::Tomb { .. } => continue 'retry,
                    Node::List { .. } => unreachable!(),
                };
                println!("set loop n:{}", n);

                let old_child_ptr = node.get_child(n);
                inode = match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Deep(inode) => inode,
                    Child::Leaf(item) if item.key.borrow() == &key => {
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };

                        match Node::set_child(&key, &value, n, op) {
                            CasRc::Ok(_) => break 'retry Ok(Some(item.value.clone())),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(_) if ws.len() == 0 => {
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };

                        match Node::leaf_to_list(&key, &value, n, op) {
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
                        println!("set loop xs:{:?}", xs);

                        let node_ptr = Node::new_subtrie(&key, &value, leaf, xs, &mut op);

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
        let _access_log = self.access_log.read().expect("fail-lock");
        let _epoch = self.generate_epoch();

        let ws = key_to_hashbits(&key);
        loop {
            let inode: &In<K, V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
            match self.do_remove(key, ws.clone(), inode) {
                CasRc::Ok(Some(old)) => break Some(old),
                CasRc::Ok(None) => break None,
                CasRc::Retry => (),
            }
        }
    }

    fn do_remove<Q>(&self, key: &Q, mut ws: Vec<u8>, inode: &In<K, V>) -> CasRc<Option<V>>
    where
        K: Clone + Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        let old: *mut Node<K, V> = inode.node.load(SeqCst);
        let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };

        let w = match ws.pop() {
            Some(w) => w,
            None => {
                let cas = self.generate_cas();
                let op = CasOp { inode, old, cas };
                return Node::remove_from_list(key, op);
            }
        };

        let (n, childs) = match node {
            Node::Trie { bmp, childs } => match hamming_distance(w, bmp.clone()) {
                Distance::Insert(_) => return CasRc::Ok(None),
                Distance::Set(n) => (n, childs),
            },
            Node::Tomb { .. } => return CasRc::Retry,
            Node::List { .. } => unreachable!(),
        };

        let old_child_ptr = childs[n].load(SeqCst);
        match unsafe { old_child_ptr.as_ref().unwrap() } {
            Child::Deep(next_inode) => match self.do_remove(key, ws, next_inode) {
                CasRc::Retry => CasRc::Retry,
                CasRc::Ok(None) => CasRc::Ok(None),
                CasRc::Ok(Some(old_value)) => {
                    let cas = self.generate_cas();
                    let mut op = CasOp { inode, old, cas };

                    let next_node = {
                        let next_node_ptr = next_inode.node.load(SeqCst);
                        op.cas.free_on_pass(gc::Mem::Node(next_node_ptr));
                        unsafe { next_node_ptr.as_ref().unwrap() }
                    };
                    Node::try_compact(n, next_node.to_tomb_item(), op);
                    CasRc::Ok(Some(old_value))
                }
            },
            Child::Leaf(item) if item.key.borrow() == key => {
                let cas = self.generate_cas();
                let op = CasOp { inode, old, cas };

                let old_value = item.value.clone();
                match Node::remove_child(w, n, op) {
                    CasRc::Retry => CasRc::Retry,
                    CasRc::Ok(_) => CasRc::Ok(Some(old_value)),
                }
            }
            Child::Leaf(_) => CasRc::Ok(None),
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

#[derive(PartialEq, Debug)]
enum Distance {
    Set(usize),    // found
    Insert(usize), // not found
}

fn hamming_distance(w: u8, bmp: [u128; 2]) -> Distance {
    let prefix = if w < 128 { 0 } else { bmp[0].count_ones() };
    let posn = 1_u128 << (w % 128);
    let mask: u128 = !(posn - 1);
    let bmp: u128 = if w < 128 { bmp[0] } else { bmp[1] };

    let (x, y) = ((bmp & mask), bmp);
    // TODO: optimize it with SSE or popcnt instructions, figure-out a way.
    let dist = (prefix + (x ^ y).count_ones()) as usize;

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

    let xs: Vec<u8> = (0..8)
        .map(|i| ((code >> (i * 8)) & SLOT_MASK) as u8)
        .collect();
    println!(
        "key_to_hashbits {:x}, {:?}",
        code,
        xs.iter()
            .map(|x| format!("{:x}", x))
            .collect::<Vec<String>>()
    );
    xs
}

fn get_from_list<K, V, Q>(key: &Q, items: &[Item<K, V>]) -> Option<V>
where
    K: Borrow<Q>,
    V: Clone,
    Q: PartialEq + ?Sized,
{
    match items.iter().find(|x| x.key.borrow() == key) {
        Some(x) => Some(x.value.clone()),
        None => None,
    }
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

fn update_into_list<K, V>(key: &K, value: &V, items: &mut Vec<Item<K, V>>) -> Option<V>
where
    K: Clone + PartialEq,
    V: Clone,
{
    match items.iter().enumerate().find(|(_, x)| &x.key == key) {
        Some((i, x)) => {
            let old_value = x.value.clone();
            items[i] = (key.clone(), value.clone()).into();
            Some(old_value)
        }
        None => {
            items.push((key.clone(), value.clone()).into());
            None
        }
    }
}

#[cfg(test)]
#[path = "map_test.rs"]
mod map_test;
