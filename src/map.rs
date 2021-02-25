use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    mem,
    ops::Deref,
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        mpsc, Arc, RwLock,
    },
    thread,
};

use crate::{
    gc::{self, gc_thread, Cas, Epoch, Reclaim},
    Result,
};

const SLOT_MASK: u32 = 0xF;

#[allow(unused_macros)]
macro_rules! print_ws {
    ($fmt:expr, $ws:expr) => {{
        let ws = $ws
            .iter()
            .map(|w| format!("{:x}", w))
            .collect::<Vec<String>>();
        println!($fmt, ws);
    }};
}

// TODO: validate() method
//   * make sure that Node::List are only at the last level.
//   * make sure that there are no, non-root, Node::Trie with empty childs.
// TODO: stats() method
//   * Count the number of In{}, Node{} and Child{}
//   * Count the number of Tomb nodes.
//   * Count the number of Nodes that are tombable.

pub struct Map<V> {
    id: usize,
    root: Arc<Root<V>>,

    epoch: Arc<AtomicU64>,
    access_log: Arc<RwLock<Vec<Arc<AtomicU64>>>>,

    handle: Option<thread::JoinHandle<()>>,
    tx: Option<mpsc::Sender<Reclaim<V>>>,
}

pub struct In<V> {
    node: AtomicPtr<Node<V>>,
}

pub enum Node<V> {
    Trie {
        bmp: u16,
        childs: Vec<AtomicPtr<Child<V>>>,
    },
    Tomb {
        item: Item<V>,
    },
    List {
        items: Vec<Item<V>>,
    },
}

pub enum Child<V> {
    Deep(In<V>),
    Leaf(Item<V>),
}

#[derive(Clone, PartialEq, Debug)]
pub struct Item<V> {
    key: u32,
    value: V,
}

pub struct Root<V> {
    root: AtomicPtr<In<V>>,
}

impl<V> Deref for Root<V> {
    type Target = AtomicPtr<In<V>>;

    fn deref(&self) -> &AtomicPtr<In<V>> {
        &self.root
    }
}

impl<V> Drop for Root<V> {
    fn drop(&mut self) {
        let inode = unsafe { Box::from_raw(self.root.load(SeqCst)) };
        Node::free(inode.node.load(SeqCst));
    }
}

impl<V> From<(u32, V)> for Item<V> {
    fn from((key, value): (u32, V)) -> Self {
        Item { key, value }
    }
}

impl<V> Item<V>
where
    V: Debug,
{
    fn print(&self, prefix: &str) {
        println!("{}Item<{:?},{:?}>", prefix, self.key, self.value);
    }
}

impl<V> In<V>
where
    V: Debug,
{
    fn print(&self, prefix: &str) {
        let node = unsafe { self.node.load(SeqCst).as_ref().unwrap() };
        node.print(prefix)
    }
}

impl<V> Drop for Map<V> {
    fn drop(&mut self) {
        {
            let access_log = self.access_log.write().expect("lock-panic");
            access_log[self.id].store(0, SeqCst);
        }

        mem::drop(self.tx.take());
        self.handle.take().unwrap().join().ok(); // TODO handle error
    }
}

impl<V> Map<V>
where
    V: 'static + Send,
{
    pub fn new() -> Map<V> {
        let root = {
            let node = Box::new(Node::Trie {
                bmp: u16::default(),
                childs: Vec::default(),
            });
            let inode = Box::new(In {
                node: AtomicPtr::new(Box::leak(node)),
            });
            Arc::new(Root {
                root: AtomicPtr::new(Box::leak(inode)),
            })
        };

        let epoch = Arc::new(AtomicU64::new(1));

        let access_log = Arc::new(RwLock::new(vec![Arc::new(AtomicU64::new(1))]));
        let (tx, rx) = mpsc::channel();
        let handle = {
            let args = (Arc::clone(&epoch), Arc::clone(&access_log));
            thread::spawn(move || gc_thread(args.0, args.1, rx))
        };

        Map {
            id: 0,
            root,

            epoch,
            access_log,

            handle: Some(handle),
            tx: Some(tx),
        }
    }

    pub fn cloned(&self) -> Map<V> {
        let id = {
            let mut access_log = self.access_log.write().expect("lock-panic");
            access_log.push(Arc::new(AtomicU64::new(1)));
            access_log.len().saturating_sub(1)
        };
        let (tx, rx) = mpsc::channel();
        let handle = {
            let args = (Arc::clone(&self.epoch), Arc::clone(&self.access_log));
            thread::spawn(move || gc_thread(args.0, args.1, rx))
        };
        Map {
            id,
            root: Arc::clone(&self.root),

            epoch: Arc::clone(&self.epoch),
            access_log: Arc::clone(&self.access_log),

            handle: Some(handle),
            tx: Some(tx),
        }
    }

    pub fn print(&self)
    where
        V: Debug,
    {
        let epoch = self.epoch.load(SeqCst);
        let guard = self.access_log.write().expect("lock-panic");
        let access_log = guard.iter().map(|e| e.load(SeqCst));
        println!("Map<{},{:?}", epoch, access_log);
        unsafe { self.root.load(SeqCst).as_ref().unwrap().print("  ") };
    }
}

impl<V> Map<V> {
    fn generate_cas(&self) -> Cas<V> {
        Cas::new(self.tx.as_ref().unwrap(), &self.epoch)
    }

    fn generate_epoch(&self) -> Epoch {
        let epoch = {
            let at = {
                let access_log = self.access_log.read().expect("lock-panic");
                Arc::clone(&access_log[self.id])
            };
            let epoch = Arc::clone(&self.epoch);
            Epoch::new(epoch, at)
        };

        self.epoch.fetch_add(1, SeqCst);

        epoch
    }
}

impl<V> Node<V> {
    fn to_tomb_item(&self) -> Option<Item<V>>
    where
        V: Clone,
    {
        match self {
            Node::Tomb { item } => Some(item.clone()),
            Node::Trie { .. } => None,
            Node::List { .. } => None,
        }
    }

    fn get_child(&self, n: usize) -> *mut Child<V> {
        match self {
            Node::Trie { childs, .. } => childs[n].load(SeqCst),
            Node::Tomb { .. } => unreachable!(),
            Node::List { .. } => unreachable!(),
        }
    }

    fn hamming_set(&mut self, w: u8) {
        match self {
            Node::Trie { bmp, .. } => {
                *bmp = *bmp | (1 << w);
            }
            Node::Tomb { .. } | Node::List { .. } => (),
        }
    }

    fn hamming_reset(&mut self, w: u8) {
        match self {
            Node::Trie { bmp, .. } => {
                *bmp = *bmp & (!(1 << w));
            }
            Node::Tomb { .. } | Node::List { .. } => (),
        }
    }

    fn print(&self, prefix: &str)
    where
        V: Debug,
    {
        match self {
            Node::Trie { bmp, childs } => {
                println!("{}Node::Trie<{:x},{}>", prefix, bmp, childs.len());
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

    fn count(&self) -> usize {
        match self {
            Node::Trie { childs, .. } => {
                let mut len = 0;
                for child in childs {
                    len += match unsafe { child.load(SeqCst).as_ref().unwrap() } {
                        Child::Leaf(_) => 1,
                        Child::Deep(inode) => {
                            unsafe { inode.node.load(SeqCst).as_ref().unwrap() }.count()
                        }
                    }
                }
                len
            }
            Node::List { items } => items.len(),
            Node::Tomb { .. } => 1,
        }
    }

    fn free(node: *mut Node<V>) {
        let node = unsafe { Box::from_raw(node) };
        match node.as_ref() {
            Node::Trie { bmp: _bmp, childs } => {
                for child in childs.iter() {
                    Child::free(child.load(SeqCst))
                }
            }
            Node::Tomb { .. } => (),
            Node::List { .. } => (),
        }
    }
}

impl<V> Node<V> {
    fn new_list(items: Vec<Item<V>>, cas: &mut Cas<V>) -> *mut Node<V>
    where
        V: Clone,
    {
        let node_ptr = Box::leak(Box::new(Node::List { items }));
        cas.free_on_fail(gc::Mem::Node(node_ptr));
        node_ptr
    }

    fn new_tomb(item: Item<V>) -> Box<Node<V>> {
        Box::new(Node::Tomb { item })
    }

    fn new_subtrie(
        key: u32,
        value: &V,
        leaf: &Item<V>,
        mut pairs: Vec<(u8, u8)>,
        op: &mut CasOp<V>,
    ) -> *mut Node<V>
    where
        V: Clone,
    {
        if pairs.len() == 0 {
            let item: Item<V> = (key.clone(), value.clone()).into();
            return Node::new_list(vec![item, leaf.clone()], &mut op.cas);
        }

        let (w1, w2) = pairs.pop().unwrap();
        // println!("new subtrie:{:x},{:x}", w1, w2);

        let node = if w1 == w2 {
            let mut node = {
                // one-child node trie, pointing to another intermediate node.
                let childs = {
                    let node = Self::new_subtrie(key, value, leaf, pairs, op);
                    vec![AtomicPtr::new(Child::new_deep(node, &mut op.cas))]
                };
                let bmp = u16::default();
                Box::new(Node::Trie { bmp, childs })
            };
            node.hamming_set(w1);
            node
        } else {
            let child1 = AtomicPtr::new(Child::new_leaf_from(key, value, &mut op.cas));
            let (bmp, childs) = (u16::default(), vec![child1]);
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
            node
        };

        let node_ptr = Box::leak(node);
        op.cas.free_on_fail(gc::Mem::Node(node_ptr));
        node_ptr
    }

    fn update_list(key: u32, value: &V, mut op: CasOp<V>) -> CasRc<Option<V>>
    where
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

    fn ins_child(key: u32, value: &V, w: u8, n: usize, mut op: CasOp<V>) -> CasRc<()>
    where
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let mut node = {
                    let mut childs: Vec<AtomicPtr<Child<V>>> = childs
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

    fn set_child(key: u32, value: &V, n: usize, mut op: CasOp<V>) -> CasRc<()>
    where
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                let node = {
                    let mut childs: Vec<AtomicPtr<Child<V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();
                    let new_child_ptr = Child::new_leaf_from(key, value, &mut op.cas);
                    childs[n] = AtomicPtr::new(new_child_ptr);

                    let bmp = bmp.clone();
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

    fn leaf_to_list(k: u32, v: &V, n: usize, mut op: CasOp<V>) -> CasRc<()>
    where
        V: Clone,
    {
        // convert a child node holding a leaf, into a interm-node pointing to node-list
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                let node = {
                    let mut childs: Vec<AtomicPtr<Child<V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();
                    let leaf = unsafe { old_child_ptr.as_ref().unwrap() }
                        .to_leaf_item()
                        .unwrap();
                    let new_child_ptr = {
                        let items = vec![(k.clone(), v.clone()).into(), leaf];
                        Child::new_deep(Node::new_list(items, &mut op.cas), &mut op.cas)
                    };

                    childs[n] = AtomicPtr::new(new_child_ptr);

                    let bmp = bmp.clone();
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

    fn set_trie_child(node: *mut Node<V>, n: usize, mut op: CasOp<V>) -> CasRc<()>
    where
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                let node = {
                    let mut childs: Vec<AtomicPtr<Child<V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();

                    childs[n] = AtomicPtr::new(Child::new_deep(node, &mut op.cas));

                    let bmp = bmp.clone();
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

    fn remove_from_list(key: u32, mut op: CasOp<V>) -> CasRc<Option<V>>
    where
        V: Clone,
    {
        let (node, ov) = match unsafe { op.old.as_ref().unwrap() } {
            Node::List { items } => match remove_from_list(key, items) {
                Some((mut items, ov)) if items.len() == 1 => {
                    let node = Node::new_tomb(items.remove(0));
                    (node, ov)
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

    fn remove_child(w: u8, n: usize, mut op: CasOp<V>) -> CasRc<Option<V>>
    where
        V: Clone,
    {
        // println!("remove_child w:{:x} n:{}", w, n);

        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                let node = {
                    let mut childs: Vec<AtomicPtr<Child<V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();
                    childs.remove(n);
                    assert!(childs.len() > 0, "unexpected num childs {}", childs.len());

                    if childs.len() == 1 {
                        let old_child_ptr = childs.remove(0).load(SeqCst);
                        match unsafe { old_child_ptr.as_ref().unwrap() }.to_leaf_item() {
                            Some(item) => {
                                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                                Node::new_tomb(item)
                            }
                            None => {
                                childs.insert(0, AtomicPtr::new(old_child_ptr));
                                let bmp = bmp.clone();
                                let mut node = Box::new(Node::Trie { bmp, childs });
                                node.hamming_reset(w);
                                node
                            }
                        }
                    } else {
                        let bmp = bmp.clone();
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

    fn try_compact(from_root: bool, inode: &In<V>, mut cas: Cas<V>) -> CasRc<()>
    where
        V: Clone,
    {
        let old = inode.node.load(SeqCst);

        let node = match unsafe { old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } if has_tomb_child(&childs) => {
                let bmp = bmp.clone();
                let childs = Node::compact_tomb_children(childs, &mut cas);
                Box::new(Node::Trie { bmp, childs })
            }
            Node::Trie { childs, .. } if childs.len() == 1 && !from_root => {
                let old_child_ptr = childs[0].load(SeqCst);
                match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Leaf(item) => {
                        let node = Node::new_tomb(item.clone());
                        cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                        node
                    }
                    Child::Deep(_) => return CasRc::Ok(()),
                }
            }
            Node::Trie { .. } => return CasRc::Ok(()),
            Node::Tomb { .. } => return CasRc::Retry,
            Node::List { items } if items.len() == 1 => Node::new_tomb(items[0].clone()),
            Node::List { .. } => return CasRc::Ok(()),
        };

        let new = Box::leak(node);

        cas.free_on_fail(gc::Mem::Node(new));
        cas.free_on_pass(gc::Mem::Node(old));
        cas.swing(&inode.node, old, new);
        CasRc::Retry
    }

    fn compact_tomb_children(
        childs: &Vec<AtomicPtr<Child<V>>>,
        cas: &mut Cas<V>,
    ) -> Vec<AtomicPtr<Child<V>>>
    where
        V: Clone,
    {
        let mut nchilds = Vec::with_capacity(childs.len());

        for child in childs.iter() {
            let old_child_ptr = child.load(SeqCst);
            match unsafe { old_child_ptr.as_ref().unwrap() } {
                Child::Leaf(_) => nchilds.push(AtomicPtr::new(old_child_ptr)),
                Child::Deep(next_inode) => {
                    let next_node_ptr = next_inode.node.load(SeqCst);
                    let next_node = unsafe { next_node_ptr.as_ref().unwrap() };
                    match next_node.to_tomb_item() {
                        Some(item) => {
                            cas.free_on_pass(gc::Mem::Node(next_node_ptr));
                            cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                            let new_ptr = Child::new_leaf(item, cas);
                            nchilds.push(AtomicPtr::new(new_ptr));
                        }
                        None => nchilds.push(AtomicPtr::new(old_child_ptr)),
                    }
                }
            }
        }

        nchilds
    }
}

impl<V> Child<V> {
    fn new_leaf_from(key: u32, value: &V, cas: &mut gc::Cas<V>) -> *mut Child<V>
    where
        V: Clone,
    {
        let (key, value) = (key.clone(), value.clone());
        let child_ptr = Box::leak(Box::new(Child::Leaf((key, value).into())));
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }

    fn new_leaf(leaf: Item<V>, cas: &mut gc::Cas<V>) -> *mut Child<V>
    where
        V: Clone,
    {
        let child_ptr = Box::leak(Box::new(Child::Leaf(leaf)));
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }

    fn new_deep(node: *mut Node<V>, cas: &mut gc::Cas<V>) -> *mut Child<V> {
        let inode = In {
            node: AtomicPtr::new(node),
        };
        let child_ptr = Box::leak(Box::new(Child::Deep(inode)));
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }

    fn to_leaf_item(&self) -> Option<Item<V>>
    where
        V: Clone,
    {
        match self {
            Child::Leaf(leaf) => Some(leaf.clone()),
            Child::Deep(_) => None,
        }
    }

    fn free(child: *mut Child<V>) {
        let child = unsafe { Box::from_raw(child) };
        match child.as_ref() {
            Child::Leaf(_item) => (),
            Child::Deep(inode) => Node::free(inode.node.load(SeqCst)),
        }
    }
}

impl<V> Map<V> {
    pub fn get(&self, key: u32) -> Option<V>
    where
        V: Clone,
    {
        let _epoch = self.generate_epoch();
        let _access_log = self.access_log.read().expect("fail-lock");

        let mut retry = 0;
        'retry: loop {
            retry += 1;

            let mut from_root = true; // count tree depth
            let mut ws = slots(key);
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            // print_ws!("get outer ws:{:?}", ws);

            loop {
                if retry > 1 {
                    // try compaction only on retry loop and only when depth is away
                    // from root, otherwise root may point to a tomb-node, which
                    // should never happen.
                    match Node::try_compact(from_root, inode, self.generate_cas()) {
                        CasRc::Retry => continue 'retry,
                        CasRc::Ok(_) => (),
                    }
                }
                from_root = false;

                let old = inode.node.load(SeqCst);
                let node = unsafe { old.as_ref().unwrap() };

                let w = match ws.pop() {
                    Some(w) => w,
                    None => match node {
                        Node::List { items } => break 'retry get_from_list(key, items),
                        Node::Tomb { item } if item.key == key => {
                            break 'retry Some(item.value.clone())
                        }
                        Node::Tomb { .. } => break 'retry None,
                        Node::Trie { .. } => unreachable!(),
                    },
                };
                // println!("get loop w:{:x}", w);

                inode = match node {
                    Node::Trie { bmp, childs } => {
                        let hd = hamming_distance(w, bmp.clone());
                        // println!("get loop bmp:{:x} {:?}", bmp, hd);
                        match hd {
                            Distance::Insert(_) => break 'retry None,
                            Distance::Set(n) => {
                                let ptr = childs[n].load(SeqCst);
                                match unsafe { ptr.as_ref().unwrap() } {
                                    Child::Deep(next_inode) => next_inode,
                                    Child::Leaf(item) if item.key == key => {
                                        break 'retry Some(item.value.clone());
                                    }
                                    Child::Leaf(_) => break 'retry None,
                                }
                            }
                        }
                    }
                    Node::List { .. } => unreachable!(),
                    Node::Tomb { item } if item.key == key => {
                        break 'retry Some(item.value.clone())
                    }
                    Node::Tomb { .. } => break 'retry None,
                }
            }
        }
    }

    pub fn set(&self, key: u32, value: V) -> Result<Option<V>>
    where
        V: Clone,
    {
        let _epoch = self.generate_epoch();
        let _access_log = self.access_log.read().expect("fail-lock");

        let mut retry = 0;
        'retry: loop {
            retry += 1;

            let mut from_root = true;
            let mut ws = slots(key);
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            // print_ws!("set try {:?}", ws);

            loop {
                if retry > 1 {
                    // try compaction only on retry loop and only when depth is away
                    // from root, otherwise root may point to a tomb-node, which
                    // should never happen.
                    match Node::try_compact(from_root, inode, self.generate_cas()) {
                        CasRc::Retry => continue 'retry,
                        CasRc::Ok(_) => (),
                    }
                }
                from_root = false;

                let old: *mut Node<V> = inode.node.load(SeqCst);
                let node: &Node<V> = unsafe { old.as_ref().unwrap() };

                let w = match ws.pop() {
                    Some(w) => w,
                    None => match node {
                        Node::Tomb { .. } => continue 'retry,
                        Node::List { .. } => {
                            let cas = self.generate_cas();
                            let op = CasOp { inode, old, cas };
                            match Node::update_list(key, &value, op) {
                                CasRc::Ok(old_value) => break 'retry Ok(old_value),
                                CasRc::Retry => continue 'retry,
                            }
                        }
                        Node::Trie { .. } => unreachable!(),
                    },
                };
                // println!("set loop w:{:x}", w);

                let n = match node {
                    Node::Trie { bmp, .. } => match hamming_distance(w, bmp.clone()) {
                        Distance::Insert(n) => {
                            // println!("set loop bmp:{:x} {}", bmp, n);
                            let cas = self.generate_cas();
                            let op = CasOp { inode, old, cas };

                            match Node::ins_child(key, &value, w, n, op) {
                                CasRc::Ok(_) => break 'retry Ok(None),
                                CasRc::Retry => continue 'retry,
                            }
                        }
                        Distance::Set(n) => n,
                    },
                    Node::Tomb { .. } => continue 'retry,
                    Node::List { .. } => unreachable!(),
                };
                // println!("set loop n:{}", n);

                let old_child_ptr = node.get_child(n);
                inode = match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Deep(inode) => inode,
                    Child::Leaf(item) if item.key == key => {
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };

                        match Node::set_child(key, &value, n, op) {
                            CasRc::Ok(_) => break 'retry Ok(Some(item.value.clone())),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(_) if ws.len() == 0 => {
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };

                        match Node::leaf_to_list(key, &value, n, op) {
                            CasRc::Ok(_) => break 'retry Ok(None),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(leaf) => {
                        let cas = self.generate_cas();
                        let mut op = CasOp { inode, old, cas };

                        let xs: Vec<(u8, u8)> = {
                            let ls = slots(leaf.key)[..ws.len()].to_vec();
                            ws.clone().into_iter().zip(ls.into_iter()).collect()
                        };
                        // println!("set loop xs:{:?}", xs);

                        let node_ptr = Node::new_subtrie(key, &value, leaf, xs, &mut op);

                        match Node::set_trie_child(node_ptr, n, op) {
                            CasRc::Ok(_) => break 'retry Ok(None),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                }
            }
        }
    }

    pub fn remove(&self, key: u32) -> Option<V>
    where
        V: Clone,
    {
        let _epoch = self.generate_epoch();
        let _access_log = self.access_log.read().expect("fail-lock");

        loop {
            let ws = slots(key);
            // print_ws!("remove try {:?}", ws);
            let inode: &In<V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
            match self.do_remove(key, ws.clone(), inode, true) {
                CasRc::Ok(Some(old)) => break Some(old),
                CasRc::Ok(None) => break None,
                CasRc::Retry => (),
            }
        }
    }

    fn do_remove(
        &self,
        key: u32,
        mut ws: Vec<u8>,
        inode: &In<V>,
        from_root: bool,
    ) -> CasRc<Option<V>>
    where
        V: Clone,
    {
        match Node::try_compact(from_root, inode, self.generate_cas()) {
            CasRc::Retry => return CasRc::Retry,
            CasRc::Ok(_) => (),
        }

        let old: *mut Node<V> = inode.node.load(SeqCst);
        let node: &Node<V> = unsafe { old.as_ref().unwrap() };

        let w = match ws.pop() {
            Some(w) => w,
            None => match node {
                Node::Tomb { .. } => return CasRc::Retry,
                Node::List { items } if items.len() == 1 => return CasRc::Retry,
                Node::List { items } => {
                    return if items.len() == 1 && !from_root {
                        CasRc::Retry
                    } else {
                        let cas = self.generate_cas();
                        let op = CasOp { inode, old, cas };
                        Node::remove_from_list(key, op)
                    }
                }
                Node::Trie { .. } => unreachable!(),
            },
        };
        // println!("do_remove w:{:x}", w);

        let (n, _bmp, childs) = match node {
            Node::Tomb { .. } => return CasRc::Retry,
            Node::Trie { bmp, childs } => match hamming_distance(w, bmp.clone()) {
                Distance::Insert(_) => return CasRc::Ok(None),
                Distance::Set(n) => (n, bmp, childs),
            },
            Node::List { .. } => unreachable!(),
        };
        // println!("do_remove n:{} bmp:{:x}", n, _bmp);

        let old_child_ptr = childs[n].load(SeqCst);
        match unsafe { old_child_ptr.as_ref().unwrap() } {
            Child::Deep(next_inode) => match self.do_remove(key, ws, next_inode, false) {
                CasRc::Retry => CasRc::Retry,
                CasRc::Ok(None) => CasRc::Ok(None),
                CasRc::Ok(Some(old_value)) => CasRc::Ok(Some(old_value)),
            },
            Child::Leaf(item) if item.key == key => {
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

    pub fn len(&self) -> usize {
        let _access_log = self.access_log.write().expect("fail-lock");

        let inode: &In<V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
        let node = unsafe { inode.node.load(SeqCst).as_ref().unwrap() };
        node.count()
    }
}

enum CasRc<T> {
    Ok(T),
    Retry,
}

struct CasOp<'a, V> {
    inode: &'a In<V>,
    old: *mut Node<V>,
    cas: Cas<'a, V>,
}

#[derive(PartialEq, Debug)]
enum Distance {
    Set(usize),    // found
    Insert(usize), // not found
}

fn hamming_distance(w: u8, bmp: u16) -> Distance {
    let posn = 1_u16 << w;
    let mask: u16 = !(posn - 1);

    let (x, y) = ((bmp & mask), bmp);
    // TODO: optimize it with SSE or popcnt instructions, figure-out a way.
    let dist = (x ^ y).count_ones() as usize;

    match bmp & posn {
        0 => Distance::Insert(dist),
        _ => Distance::Set(dist),
    }
}

// TODO: Can we make this to use a generic hash function ?
pub fn key_to_hashbits<K>(key: &K) -> Vec<u8>
where
    K: Hash + ?Sized,
{
    use fasthash::city::crc;

    let mut hasher = crc::Hasher128::default();
    key.hash(&mut hasher);
    let code: u64 = hasher.finish();
    let code: u32 = (((code >> 32) ^ code) & 0xFFFFFFFF) as u32;
    slots(code)
}

#[inline]
fn slots(key: u32) -> Vec<u8> {
    (0..8)
        .map(|i| ((key >> (i * 4)) & SLOT_MASK) as u8)
        .collect()
}

fn get_from_list<V>(key: u32, items: &[Item<V>]) -> Option<V>
where
    V: Clone,
{
    match items.iter().find(|x| x.key == key) {
        Some(x) => Some(x.value.clone()),
        None => None,
    }
}

fn remove_from_list<V>(key: u32, items: &[Item<V>]) -> Option<(Vec<Item<V>>, V)>
where
    V: Clone,
{
    let res = items.iter().enumerate().find(|(_, x)| x.key == key);

    match res {
        Some((i, x)) => {
            let mut xs = items[..i].to_vec();
            xs.extend_from_slice(&items[i + 1..]);
            Some((xs, x.value.clone()))
        }
        None => None,
    }
}

fn update_into_list<V>(key: u32, value: &V, items: &mut Vec<Item<V>>) -> Option<V>
where
    V: Clone,
{
    match items.iter().enumerate().find(|(_, x)| x.key == key) {
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

fn has_tomb_child<V>(childs: &[AtomicPtr<Child<V>>]) -> bool {
    childs.iter().any(
        |child| match unsafe { child.load(SeqCst).as_ref().unwrap() } {
            Child::Leaf(_) => false,
            Child::Deep(inode) => {
                let node = unsafe { inode.node.load(SeqCst).as_ref().unwrap() };
                match node {
                    Node::Tomb { .. } => true,
                    _ => false,
                }
            }
        },
    )
}

#[cfg(test)]
#[path = "map_test.rs"]
mod map_test;
