use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    ops::Deref,
    sync::{
        self,
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        Arc, RwLock,
    },
    thread, time,
};

use crate::gc::{self, Cas, Epoch};

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

macro_rules! gc_epoch {
    ($log:expr, $epoch:expr) => {{
        let mut gc_epoch = u64::MAX;
        for e in $log.iter() {
            let thread_epoch = e.load(SeqCst);
            let thread_epoch = if thread_epoch & gc::ENTER_MASK == 0 {
                $epoch
            } else {
                thread_epoch & gc::EPOCH_MASK
            };
            gc_epoch = u64::min(gc_epoch, thread_epoch);
        }
        gc_epoch
    }};
}

// TODO: validate() method
//   * make sure that Node::List are only at the last level.
//   * make sure that there are no, non-root, Node::Trie with empty childs.
//   * one the root In{} and Node{} is allocated it should never be deallocated.
// TODO: stats() method
//   * Count the number of In{}, Node{} and Child{}
//   * Count the number of Tomb nodes.
//   * Count the number of Nodes that are tombable.
// TODO: compact logic
//   * To be called external to the library.
//   * Check for Tomb nodes.
//   * Compact trie-childs to capacity == length.
//   * Compact list-items to capacity == length.
// TODO:
//   * review `_ => ` catch all arms.
//   * review expect() and unwrap() calls
//   * review `as` type-casts.

type RGuard<'a> = sync::RwLockReadGuard<'a, Vec<Arc<AtomicU64>>>;

pub struct Map<V> {
    id: usize,
    root: Arc<Root<V>>,

    epoch: Arc<AtomicU64>,
    access_log: Arc<RwLock<Vec<Arc<AtomicU64>>>>,
    cas: gc::Cas<V>,
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
    value: Option<V>,
}

impl<V> Default for Item<V> {
    fn default() -> Self {
        Item {
            key: 0,
            value: None,
        }
    }
}

impl<V> Default for Child<V> {
    fn default() -> Self {
        Child::Leaf(Item::default())
    }
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
        Node::dropped(inode.node.load(SeqCst));
    }
}

impl<V> From<(u32, V)> for Item<V> {
    fn from((key, value): (u32, V)) -> Self {
        Item {
            key,
            value: Some(value),
        }
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
        while self.cas.has_reclaims() {
            let epoch = self.epoch.load(SeqCst);
            let gc_epoch = {
                let access_log = self.access_log.read().expect("fail-lock");
                gc_epoch!(access_log, epoch)
            };
            if gc_epoch == 0 || gc_epoch == u64::MAX {
                // force collect, either all clones have been dropped or there was none.
                self.cas.garbage_collect(u64::MAX)
            } else if gc_epoch < u64::MAX {
                self.cas.garbage_collect(gc_epoch)
            }
            thread::sleep(time::Duration::from_millis(10)); // TODO exponential backoff
        }
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

        let access_log = Arc::new(RwLock::new(vec![Arc::new(AtomicU64::new(1))]));
        Map {
            id: 0,
            root,

            epoch: Arc::new(AtomicU64::new(1)),
            access_log,
            cas: gc::Cas::new(),
        }
    }

    pub fn cloned(&self) -> Map<V> {
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
            cas: gc::Cas::new(),
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
    #[inline]
    fn get_child(&self, n: usize) -> *mut Child<V> {
        match self {
            Node::Trie { childs, .. } => childs[n].load(SeqCst),
            Node::Tomb { .. } => unreachable!(),
            Node::List { .. } => unreachable!(),
        }
    }

    #[inline]
    fn hamming_set(&mut self, w: u8) {
        match self {
            Node::Trie { bmp, .. } => {
                *bmp = *bmp | (1 << w);
            }
            Node::Tomb { .. } | Node::List { .. } => (),
        }
    }

    #[inline]
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

    fn dropped(node: *mut Node<V>) {
        let node = unsafe { Box::from_raw(node) };
        match node.as_ref() {
            Node::Trie { bmp: _bmp, childs } => {
                for child in childs.iter() {
                    Child::dropped(child.load(SeqCst))
                }
            }
            Node::Tomb { .. } => (),
            Node::List { .. } => (),
        }
    }
}

impl<V> Node<V> {
    fn new_bi_list(k: u32, v: &V, leaf: &Item<V>, cas: &mut Cas<V>) -> *mut Node<V>
    where
        V: Clone,
    {
        let mut node = cas.alloc_node('l');
        match node.as_mut() {
            Node::List { items } => {
                items.clear();
                unsafe { items.set_len(2) };
                items[0] = (k, v.clone()).into();
                items[1].clone_from(leaf);
            }
            _ => unreachable!(),
        };
        let node_ptr = Box::leak(node);
        cas.free_on_fail(gc::Mem::Node(node_ptr));
        node_ptr
    }

    fn new_list_without(olds: &[Item<V>], i: usize, cas: &mut Cas<V>) -> *mut Node<V>
    where
        V: Clone,
    {
        let mut node = cas.alloc_node('l');
        match node.as_mut() {
            Node::List { items } => {
                items.clear();
                items.extend_from_slice(&olds[..i]);
                // skip i
                items.extend_from_slice(&olds[i + 1..]);
            }
            _ => unreachable!(),
        }
        let node_ptr = Box::leak(node);
        cas.free_on_fail(gc::Mem::Node(node_ptr));
        node_ptr
    }

    fn new_tomb(nitem: &Item<V>, cas: &mut Cas<V>) -> *mut Node<V>
    where
        V: Clone,
    {
        let mut node = cas.alloc_node('b');
        match node.as_mut() {
            Node::Tomb { item } => item.clone_from(nitem),
            _ => unreachable!(),
        }
        Box::leak(node)
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
            return Node::new_bi_list(key, value, leaf, op.cas);
        }

        let (w1, w2) = pairs.pop().unwrap();
        // println!("new subtrie:{:x},{:x}", w1, w2);

        let node = if w1 == w2 {
            let mut node = {
                // one-child node trie, pointing to another intermediate node.
                let childs = {
                    let node = Self::new_subtrie(key, value, leaf, pairs, op);
                    vec![AtomicPtr::new(Child::new_deep(node, op.cas))]
                };
                let bmp = u16::default();
                Box::new(Node::Trie { bmp, childs })
            };
            node.hamming_set(w1);
            node
        } else {
            let child1 = AtomicPtr::new(Child::new_leaf_from(key, value, op.cas));
            let (bmp, childs) = (u16::default(), vec![child1]);
            let mut node = Node::Trie { bmp, childs };
            node.hamming_set(w1);

            let mut node = match node {
                Node::Trie { bmp, mut childs } => {
                    let n = match hamming_distance(w2, bmp.clone()) {
                        Distance::Insert(n) => n,
                        Distance::Set(_) => unreachable!(),
                    };
                    childs.insert(n, AtomicPtr::new(Child::new_leaf(leaf, op.cas)));
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

    fn update_list(key: u32, value: &V, op: CasOp<V>) -> CasRc<Option<V>>
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
                if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
                    CasRc::Ok(old_value)
                } else {
                    CasRc::Retry
                }
            }
            Node::Tomb { .. } => CasRc::Retry,
            Node::Trie { .. } => unreachable!(),
        }
    }

    fn ins_child(key: u32, value: &V, w: u8, n: usize, op: CasOp<V>) -> CasRc<()>
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
                    let new_child_ptr = Child::new_leaf_from(key, value, op.cas);
                    childs.insert(n, AtomicPtr::new(new_child_ptr));

                    let bmp = bmp.clone();
                    Box::new(Node::Trie { bmp, childs })
                };
                node.hamming_set(w);

                let new = Box::leak(node);

                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
            Node::Tomb { .. } => unreachable!(),
        }
    }

    fn set_child(key: u32, value: &V, n: usize, op: CasOp<V>) -> CasRc<()>
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
                    let new_child_ptr = Child::new_leaf_from(key, value, op.cas);
                    childs[n] = AtomicPtr::new(new_child_ptr);

                    let bmp = bmp.clone();
                    Box::new(Node::Trie { bmp, childs })
                };

                let new = Box::leak(node);

                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
            Node::Tomb { .. } => unreachable!(),
        }
    }

    fn leaf_to_list(k: u32, v: &V, n: usize, op: CasOp<V>) -> CasRc<()>
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

                    let new_child_ptr = match unsafe { old_child_ptr.as_ref().unwrap() } {
                        Child::Leaf(item) => {
                            let node = Node::new_bi_list(k, v, item, op.cas);
                            Child::new_deep(node, op.cas)
                        }
                        Child::Deep(_) => unreachable!(),
                    };

                    childs[n] = AtomicPtr::new(new_child_ptr);

                    let bmp = bmp.clone();
                    Box::new(Node::Trie { bmp, childs })
                };

                let new = Box::leak(node);

                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
            Node::Tomb { .. } => unreachable!(),
        }
    }

    fn set_trie_child(node: *mut Node<V>, n: usize, op: CasOp<V>) -> CasRc<()>
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

                    childs[n] = AtomicPtr::new(Child::new_deep(node, op.cas));

                    let bmp = bmp.clone();
                    Box::new(Node::Trie { bmp, childs })
                };

                let new = Box::leak(node);

                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
                    CasRc::Ok(())
                } else {
                    CasRc::Retry
                }
            }
            Node::List { .. } => unreachable!(),
            Node::Tomb { .. } => unreachable!(),
        }
    }

    fn remove_from_list(key: u32, op: CasOp<V>) -> CasRc<Option<V>>
    where
        V: Clone,
    {
        let (new, ov) = match unsafe { op.old.as_ref().unwrap() } {
            Node::List { items } => {
                let res = items.iter().enumerate().find(|(_, x)| x.key == key);
                match res {
                    Some((i, _)) if items.len() == 2 => {
                        let j = [1, 0][i];
                        let ov = items[i].value.as_ref().cloned().unwrap();
                        (Node::new_tomb(&items[j], op.cas), ov)
                    }
                    Some((_, _)) if items.len() == 1 => unreachable!(),
                    Some((i, _)) => {
                        let ov = items[i].value.as_ref().cloned().unwrap();
                        (Node::new_list_without(items, i, op.cas), ov)
                    }
                    None => return CasRc::Ok(None),
                }
            }
            Node::Tomb { .. } => return CasRc::Retry,
            Node::Trie { .. } => unreachable!(),
        };

        op.cas.free_on_pass(gc::Mem::Node(op.old));
        op.cas.free_on_fail(gc::Mem::Node(new));
        if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
            CasRc::Ok(Some(ov))
        } else {
            CasRc::Retry
        }
    }

    fn remove_child(w: u8, n: usize, op: CasOp<V>) -> CasRc<Option<V>>
    where
        V: Clone,
    {
        // println!("remove_child w:{:x} n:{}", w, n);

        match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } => {
                let old_child_ptr = childs[n].load(SeqCst);
                let new = {
                    let mut childs: Vec<AtomicPtr<Child<V>>> = childs
                        .iter()
                        .map(|c| AtomicPtr::new(c.load(SeqCst)))
                        .collect();
                    childs.remove(n);
                    assert!(childs.len() > 0, "unexpected num childs {}", childs.len());

                    if childs.len() == 1 {
                        let old_child_ptr = childs.remove(0).load(SeqCst);
                        match unsafe { old_child_ptr.as_ref().unwrap() } {
                            Child::Leaf(item) => {
                                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                                Node::new_tomb(item, op.cas)
                            }
                            Child::Deep(_) => {
                                childs.insert(0, AtomicPtr::new(old_child_ptr));
                                let bmp = bmp.clone();
                                let mut node = Box::new(Node::Trie { bmp, childs });
                                node.hamming_reset(w);
                                Box::leak(node)
                            }
                        }
                    } else {
                        let bmp = bmp.clone();
                        let mut node = Box::new(Node::Trie { bmp, childs });
                        node.hamming_reset(w);
                        Box::leak(node)
                    }
                };

                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                op.cas.free_on_fail(gc::Mem::Node(new));
                op.cas.free_on_pass(gc::Mem::Node(op.old));
                if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
                    CasRc::Ok(None)
                } else {
                    CasRc::Retry
                }
            }
            Node::Tomb { .. } => unreachable!(),
            Node::List { .. } => unreachable!(),
        }
    }

    fn try_compact(fr: bool, inode: &In<V>, cas: &mut Cas<V>, epoch: &Arc<AtomicU64>) -> CasRc<()>
    where
        V: Clone,
    {
        let old = inode.node.load(SeqCst);

        let new = match unsafe { old.as_ref().unwrap() } {
            Node::Trie { bmp, childs } if has_tomb_child(&childs) => {
                let bmp = bmp.clone();
                let childs = Node::compact_tomb_children(childs, cas);
                Box::leak(Box::new(Node::Trie { bmp, childs }))
            }
            Node::Trie { childs, .. } if childs.len() == 1 && !fr => {
                let old_child_ptr = childs[0].load(SeqCst);
                match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Leaf(item) => {
                        let new = Node::new_tomb(item, cas);
                        cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                        new
                    }
                    Child::Deep(_) => return CasRc::Ok(()),
                }
            }
            Node::Trie { .. } => return CasRc::Ok(()),
            Node::Tomb { .. } => return CasRc::Retry,
            Node::List { items } if items.len() == 1 => Node::new_tomb(&items[0], cas),
            Node::List { .. } => return CasRc::Ok(()),
        };

        cas.free_on_fail(gc::Mem::Node(new));
        cas.free_on_pass(gc::Mem::Node(old));
        cas.swing(epoch, &inode.node, old, new);
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
                    match unsafe { next_node_ptr.as_ref().unwrap() } {
                        Node::Tomb { item } => {
                            cas.free_on_pass(gc::Mem::Node(next_node_ptr));
                            cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                            let new_ptr = Child::new_leaf(item, cas);
                            nchilds.push(AtomicPtr::new(new_ptr));
                        }
                        _ => nchilds.push(AtomicPtr::new(old_child_ptr)),
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
        let mut child = cas.alloc_child();

        *child = Child::Leaf((key, value.clone()).into());

        let child_ptr = Box::leak(child);
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }

    fn new_leaf(leaf: &Item<V>, cas: &mut gc::Cas<V>) -> *mut Child<V>
    where
        V: Clone,
    {
        let mut child = cas.alloc_child();

        *child = Child::Leaf(leaf.clone());

        let child_ptr = Box::leak(child);
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

    fn dropped(child: *mut Child<V>) {
        let child = unsafe { Box::from_raw(child) };
        match child.as_ref() {
            Child::Leaf(_item) => (),
            Child::Deep(inode) => Node::dropped(inode.node.load(SeqCst)),
        }
    }
}

impl<V> Map<V> {
    pub fn get(&mut self, key: u32) -> Option<V>
    where
        V: Clone,
    {
        let epoch = self.epoch.load(SeqCst);
        let (gc_epoch, res) = {
            let (access_log, res) = self.do_get(key);
            (gc_epoch!(access_log, epoch), res)
        };
        if gc_epoch < u64::MAX {
            self.cas.garbage_collect(gc_epoch)
        }
        res
    }

    fn do_get(&mut self, key: u32) -> (RGuard, Option<V>)
    where
        V: Clone,
    {
        let _epoch = self.generate_epoch();
        let access_log = self.access_log.read().expect("fail-lock");

        let mut retry = 0;
        let ws = slots(key);
        let res = 'retry: loop {
            retry += 1;

            let mut fr = true; // count tree depth
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            let mut wss = &ws[..];
            // print_ws!("get outer ws:{:?}", wss);

            loop {
                if retry > 1 {
                    // try compaction only on retry loop and only when depth is away
                    // from root, otherwise root may point to a tomb-node, which
                    // should never happen.
                    match Node::try_compact(fr, inode, &mut self.cas, &self.epoch) {
                        CasRc::Retry => continue 'retry,
                        CasRc::Ok(_) => (),
                    }
                }
                fr = false;

                let old = inode.node.load(SeqCst);
                let node = unsafe { old.as_ref().unwrap() };

                let w = match wss.last() {
                    Some(w) => *w,
                    None => match node {
                        Node::List { items } => break 'retry get_from_list(key, items),
                        Node::Tomb { item } if item.key == key => {
                            break 'retry Some(item.value.as_ref().cloned().unwrap())
                        }
                        Node::Tomb { .. } => break 'retry None,
                        Node::Trie { .. } => unreachable!(),
                    },
                };
                wss = &wss[..wss.len() - 1];
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
                                        let ov = item.value.as_ref().cloned().unwrap();
                                        break 'retry Some(ov);
                                    }
                                    Child::Leaf(_) => break 'retry None,
                                }
                            }
                        }
                    }
                    Node::List { .. } => unreachable!(),
                    Node::Tomb { item } if item.key == key => {
                        break 'retry Some(item.value.as_ref().cloned().unwrap())
                    }
                    Node::Tomb { .. } => break 'retry None,
                }
            }
        };
        (access_log, res)
    }

    pub fn set(&mut self, key: u32, value: V) -> Option<V>
    where
        V: Clone,
    {
        let epoch = self.epoch.load(SeqCst);
        let (gc_epoch, res) = {
            let (access_log, res) = self.do_set(key, value);
            (gc_epoch!(access_log, epoch), res)
        };
        if gc_epoch < u64::MAX {
            self.cas.garbage_collect(gc_epoch)
        }
        res
    }

    fn do_set(&mut self, key: u32, value: V) -> (RGuard, Option<V>)
    where
        V: Clone,
    {
        let _epoch = self.generate_epoch();
        let access_log = self.access_log.read().expect("fail-lock");

        let mut retry = 0;
        let ws = slots(key);
        let res = 'retry: loop {
            retry += 1;

            let mut fr = true;
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            let mut wss = &ws[..];
            // print_ws!("set try {:?}", ws);

            loop {
                if retry > 1 {
                    // try compaction only on retry loop and only when depth is away
                    // from root, otherwise root may point to a tomb-node, which
                    // should never happen.
                    match Node::try_compact(fr, inode, &mut self.cas, &self.epoch) {
                        CasRc::Retry => continue 'retry,
                        CasRc::Ok(_) => (),
                    }
                }
                fr = false;

                let old: *mut Node<V> = inode.node.load(SeqCst);
                let node: &Node<V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.last() {
                    Some(w) => *w,
                    None => match node {
                        Node::Tomb { .. } => continue 'retry,
                        Node::List { .. } => {
                            let op = CasOp {
                                epoch: &self.epoch,
                                inode,
                                old,
                                cas: &mut self.cas,
                            };

                            match Node::update_list(key, &value, op) {
                                CasRc::Ok(old_value) => break 'retry old_value,
                                CasRc::Retry => continue 'retry,
                            }
                        }
                        Node::Trie { .. } => unreachable!(),
                    },
                };
                wss = &wss[..wss.len() - 1];
                // println!("set loop w:{:x}", w);

                let n = match node {
                    Node::Trie { bmp, .. } => match hamming_distance(w, bmp.clone()) {
                        Distance::Insert(n) => {
                            // println!("set loop bmp:{:x} {}", bmp, n);
                            let op = CasOp {
                                epoch: &self.epoch,
                                inode,
                                old,
                                cas: &mut self.cas,
                            };

                            match Node::ins_child(key, &value, w, n, op) {
                                CasRc::Ok(_) => break 'retry None,
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
                        let op = CasOp {
                            epoch: &self.epoch,
                            inode,
                            old,
                            cas: &mut self.cas,
                        };

                        match Node::set_child(key, &value, n, op) {
                            CasRc::Ok(_) => {
                                break 'retry Some(item.value.as_ref().cloned().unwrap());
                            }
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(_) if wss.len() == 0 => {
                        let op = CasOp {
                            epoch: &self.epoch,
                            inode,
                            old,
                            cas: &mut self.cas,
                        };

                        match Node::leaf_to_list(key, &value, n, op) {
                            CasRc::Ok(_) => break 'retry None,
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(leaf) => {
                        let mut op = CasOp {
                            epoch: &self.epoch,
                            inode,
                            old,
                            cas: &mut self.cas,
                        };

                        let xs: Vec<(u8, u8)> = {
                            let ls = slots(leaf.key)[..wss.len()].to_vec();
                            wss.to_vec().into_iter().zip(ls.into_iter()).collect()
                        };
                        // println!("set loop xs:{:?}", xs);

                        let node_ptr = Node::new_subtrie(key, &value, leaf, xs, &mut op);

                        match Node::set_trie_child(node_ptr, n, op) {
                            CasRc::Ok(_) => break 'retry None,
                            CasRc::Retry => continue 'retry,
                        }
                    }
                }
            }
        };
        (access_log, res)
    }

    pub fn remove(&mut self, key: u32) -> Option<V>
    where
        V: Clone,
    {
        let epoch = self.epoch.load(SeqCst);
        let (gc_epoch, res) = {
            let (access_log, res) = self.do_remove(key);
            (gc_epoch!(access_log, epoch), res)
        };
        if gc_epoch < u64::MAX {
            self.cas.garbage_collect(gc_epoch)
        }
        res
    }

    fn do_remove(&mut self, key: u32) -> (RGuard, Option<V>)
    where
        V: Clone,
    {
        let _epoch = self.generate_epoch();
        let access_log = self.access_log.read().expect("fail-lock");

        let ws = slots(key);
        let res = 'retry: loop {
            let mut fr = true;
            let mut inode: &In<V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            let mut wss = &ws[..];
            // print_ws!("remove try {:?}", wss);

            loop {
                match Node::try_compact(fr, inode, &mut self.cas, &self.epoch) {
                    CasRc::Retry => continue 'retry,
                    CasRc::Ok(_) => (),
                }
                fr = false;

                let old: *mut Node<V> = inode.node.load(SeqCst);
                let node: &Node<V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.last() {
                    Some(w) => *w,
                    None => match node {
                        Node::Tomb { .. } => continue 'retry,
                        Node::List { items } => {
                            if items.len() == 1 && !fr {
                                continue 'retry;
                            } else {
                                let op = CasOp {
                                    epoch: &self.epoch,
                                    inode,
                                    old,
                                    cas: &mut self.cas,
                                };
                                match Node::remove_from_list(key, op) {
                                    CasRc::Ok(old_value) => break 'retry old_value,
                                    CasRc::Retry => continue 'retry,
                                }
                            }
                        }
                        Node::Trie { .. } => unreachable!(),
                    },
                };
                wss = &wss[..wss.len() - 1];
                // println!("do_remove w:{:x}", w);

                let (n, _bmp, childs) = match node {
                    Node::Tomb { .. } => continue 'retry,
                    Node::Trie { bmp, childs } => match hamming_distance(w, bmp.clone()) {
                        Distance::Insert(_) => break 'retry None,
                        Distance::Set(n) => (n, bmp, childs),
                    },
                    Node::List { .. } => unreachable!(),
                };
                // println!("do_remove n:{} bmp:{:x}", n, _bmp);

                let old_child_ptr = childs[n].load(SeqCst);
                inode = match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Deep(next_inode) => next_inode,
                    Child::Leaf(item) if item.key == key => {
                        let op = CasOp {
                            epoch: &self.epoch,
                            inode,
                            old,
                            cas: &mut self.cas,
                        };
                        let old_value = item.value.as_ref().cloned().unwrap();
                        match Node::remove_child(w, n, op) {
                            CasRc::Retry => continue 'retry,
                            CasRc::Ok(_) => break 'retry Some(old_value),
                        }
                    }
                    Child::Leaf(_) => break 'retry None,
                }
            }
        };
        (access_log, res)
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
    epoch: &'a Arc<AtomicU64>,
    inode: &'a In<V>,
    old: *mut Node<V>,
    cas: &'a mut Cas<V>,
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
pub fn key_to_hashbits<K>(key: &K) -> [u8; 8]
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
fn slots(key: u32) -> [u8; 8] {
    let mut arr = [0_u8; 8];
    for i in 0..8 {
        arr[i] = ((key >> (i * 4)) & SLOT_MASK) as u8;
    }
    arr
}

fn get_from_list<V>(key: u32, items: &[Item<V>]) -> Option<V>
where
    V: Clone,
{
    match items.iter().find(|x| x.key == key) {
        Some(x) => Some(x.value.as_ref().cloned().unwrap()),
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
            Some(old_value.unwrap())
        }
        None => {
            items.push((key, value.clone()).into());
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
