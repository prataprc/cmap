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

macro_rules! generate_op {
    ($this:expr, $inode:expr, $old:expr) => {
        CasOp {
            epoch: &$this.epoch,
            inode: $inode,
            old: $old,
            cas: &mut $this.cas,
        }
    };
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
            } else {
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

        epoch
    }
}

impl<V> Child<V> {
    fn new_leaf(leaf: Item<V>, cas: &mut gc::Cas<V>) -> *mut Child<V>
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

        let mut child = cas.alloc_child();

        *child = Child::Deep(inode);

        let child_ptr = Box::leak(child);
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }

    fn is_tomb_node(&self) -> bool {
        match self {
            Child::Leaf(_) => false,
            Child::Deep(inode) => {
                let node = unsafe { inode.node.load(SeqCst).as_ref().unwrap() };
                match node {
                    Node::Tomb { .. } => true,
                    _ => false,
                }
            }
        }
    }

    fn dropped(child: *mut Child<V>) {
        let child = unsafe { Box::from_raw(child) };
        match child.as_ref() {
            Child::Leaf(_item) => (),
            Child::Deep(inode) => Node::dropped(inode.node.load(SeqCst)),
        }
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

    fn get_value(&self, key: u32) -> Option<V>
    where
        V: Clone,
    {
        match self {
            Node::List { items } => get_from_list(key, items),
            Node::Tomb { item } if item.key == key => {
                let value = item.value.as_ref().cloned().unwrap();
                Some(value)
            }
            Node::Tomb { .. } => None,
            Node::Trie { .. } => None,
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
    fn new_bi_list(item: Item<V>, leaf: &Item<V>, cas: &mut Cas<V>) -> *mut Node<V>
    where
        V: Clone,
    {
        let mut node = cas.alloc_node('l');

        match node.as_mut() {
            Node::List { items } => {
                items.clear();
                unsafe { items.set_len(2) }; // **IMPORTANT**
                items[0] = item;
                items[1].clone_from(leaf);
            }
            _ => unreachable!(),
        };

        let node_ptr = Box::leak(node);
        cas.free_on_fail(gc::Mem::Node(node_ptr));
        node_ptr
    }

    fn new_list_with(olds: &[Item<V>], item: Item<V>, cas: &mut Cas<V>) -> (*mut Node<V>, Option<V>)
    where
        V: Clone,
    {
        let mut node = cas.alloc_node('l');

        let old_value = match node.as_mut() {
            Node::List { items } => {
                items.clear();
                items.extend_from_slice(olds);
                update_into_list(item, items)
            }
            _ => unreachable!(),
        };

        let node_ptr = Box::leak(node);
        cas.free_on_fail(gc::Mem::Node(node_ptr));
        (node_ptr, old_value)
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
                items.extend_from_slice(&olds[i + 1..]); // skip i
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

        let node_ptr = Box::leak(node);
        cas.free_on_fail(gc::Mem::Node(node_ptr));
        node_ptr
    }

    fn new_subtrie(
        item: Item<V>,
        leaf: &Item<V>,
        pairs: &[(u8, u8)],
        op: &mut CasOp<V>,
    ) -> *mut Node<V>
    where
        V: Clone,
    {
        let node_ptr = match pairs.first() {
            None => Node::new_bi_list(item, leaf, op.cas),
            Some((w1, w2)) if *w1 == *w2 => {
                // println!("new subtrie:{:x},{:x}", w1, w2);
                let mut node = op.cas.alloc_node('t');

                match node.as_mut() {
                    Node::Trie { childs, .. } => {
                        let node = Self::new_subtrie(item, leaf, &pairs[1..], op);

                        childs.clear();
                        childs.insert(0, AtomicPtr::new(Child::new_deep(node, op.cas)));
                    }
                    _ => unreachable!(),
                };
                node.hamming_set(*w1);
                Box::leak(node)
            }
            Some((w1, w2)) => {
                // println!("new subtrie:{:x},{:x}", w1, w2);
                let mut node = op.cas.alloc_node('t');

                match node.as_mut() {
                    Node::Trie { childs, .. } => {
                        childs.insert(0, AtomicPtr::new(Child::new_leaf(item, op.cas)));
                    }
                    _ => unreachable!(),
                }
                node.hamming_set(*w1);

                match node.as_mut() {
                    Node::Trie { bmp, childs } => {
                        let n = match hamming_distance(*w2, bmp.clone()) {
                            Distance::Insert(n) => n,
                            Distance::Set(_) => unreachable!(),
                        };
                        let leaf = leaf.clone();
                        childs.insert(n, AtomicPtr::new(Child::new_leaf(leaf, op.cas)));
                    }
                    _ => unreachable!(),
                };
                node.hamming_set(*w2);
                Box::leak(node)
            }
        };

        op.cas.free_on_fail(gc::Mem::Node(node_ptr));
        node_ptr
    }

    fn trie_copy_from(&mut self, node: &Node<V>) {
        let (old_bmp, old_childs) = match node {
            Node::Trie {
                bmp: old_bmp,
                childs: old_childs,
            } => (old_bmp, old_childs),
            _ => unreachable!(),
        };

        match self {
            Node::Trie { bmp, childs } => {
                *bmp = *old_bmp;
                childs.clear();
                for c in old_childs.iter() {
                    childs.push(AtomicPtr::new(c.load(SeqCst)))
                }
            }
            _ => unreachable!(),
        }
    }

    fn update_list(key: u32, value: &V, op: CasOp<V>) -> CasRc<Option<V>>
    where
        V: Clone,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::List { items: olds } => {
                let item: Item<V> = (key, value.clone()).into();
                let (new, old_value) = Node::new_list_with(olds, item, op.cas);

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

    fn ins_child(leaf: Item<V>, w: u8, n: usize, op: CasOp<V>) -> CasRc<()>
    where
        V: Clone,
    {
        let new_child_ptr = Child::new_leaf(leaf, op.cas);

        let mut node = op.cas.alloc_node('t');
        node.trie_copy_from(unsafe { op.old.as_ref().unwrap() });

        match node.as_mut() {
            Node::Trie { childs, .. } => childs.insert(n, AtomicPtr::new(new_child_ptr)),
            _ => unreachable!(),
        }
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

    fn set_child(leaf: Item<V>, n: usize, op: CasOp<V>) -> CasRc<()>
    where
        V: Clone,
    {
        let old_child_ptr = match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { childs, .. } => childs[n].load(SeqCst),
            _ => unreachable!(),
        };

        let new_child_ptr = Child::new_leaf(leaf, op.cas);

        let mut node = op.cas.alloc_node('t');
        node.trie_copy_from(unsafe { op.old.as_ref().unwrap() });

        match node.as_mut() {
            Node::Trie { childs, .. } => childs[n] = AtomicPtr::new(new_child_ptr),
            _ => unreachable!(),
        }

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

    fn leaf_to_list(key: u32, value: &V, n: usize, op: CasOp<V>) -> CasRc<()>
    where
        V: Clone,
    {
        // convert a child node holding a leaf, into a interm-node pointing to node-list
        let old_child_ptr = match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { childs, .. } => childs[n].load(SeqCst),
            _ => unreachable!(),
        };
        let old_child = unsafe { old_child_ptr.as_ref().unwrap() };

        let mut node = op.cas.alloc_node('t');
        node.trie_copy_from(unsafe { op.old.as_ref().unwrap() });

        match node.as_mut() {
            Node::Trie { childs, .. } => {
                let new_child_ptr = match old_child {
                    Child::Leaf(leaf) => {
                        let item: Item<V> = (key, value.clone()).into();
                        let node = Node::new_bi_list(item, leaf, op.cas);
                        Child::new_deep(node, op.cas)
                    }
                    Child::Deep(_) => unreachable!(),
                };
                childs[n] = AtomicPtr::new(new_child_ptr);
            }
            _ => unreachable!(),
        }

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

    fn set_trie_child(node: *mut Node<V>, n: usize, op: CasOp<V>) -> CasRc<()>
    where
        V: Clone,
    {
        let old_child_ptr = match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { childs, .. } => childs[n].load(SeqCst),
            _ => unreachable!(),
        };

        let mut curr_node = op.cas.alloc_node('t');
        curr_node.trie_copy_from(unsafe { op.old.as_ref().unwrap() });

        match curr_node.as_mut() {
            Node::Trie { childs, .. } => {
                childs[n] = AtomicPtr::new(Child::new_deep(node, op.cas));
            }
            _ => unreachable!(),
        }

        let new = Box::leak(curr_node);

        op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
        op.cas.free_on_fail(gc::Mem::Node(new));
        op.cas.free_on_pass(gc::Mem::Node(op.old));
        if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
            CasRc::Ok(())
        } else {
            CasRc::Retry
        }
    }

    fn remove_from_list(n: usize, op: CasOp<V>) -> (bool, CasRc<Option<V>>)
    where
        V: Clone,
    {
        let (compact, new, ov) = match unsafe { op.old.as_ref().unwrap() } {
            Node::List { items } if items.len() == 1 => {
                let ov = items[n].value.as_ref().cloned().unwrap();
                (true, Node::new_list_without(items, n, op.cas), ov)
            }
            Node::List { items } if items.len() == 2 => {
                let j = [1, 0][n];
                let ov = items[n].value.as_ref().cloned().unwrap();
                (true, Node::new_tomb(&items[j], op.cas), ov)
            }
            Node::List { items } if items.len() > 0 => {
                let ov = items[n].value.as_ref().cloned().unwrap();
                (false, Node::new_list_without(items, n, op.cas), ov)
            }
            Node::List { .. } => unreachable!(),
            Node::Tomb { .. } => unreachable!(),
            Node::Trie { .. } => unreachable!(),
        };

        op.cas.free_on_pass(gc::Mem::Node(op.old));
        if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
            (compact, CasRc::Ok(Some(ov)))
        } else {
            (compact, CasRc::Retry)
        }
    }

    fn remove_child(w: u8, n: usize, op: CasOp<V>) -> (bool, CasRc<Option<V>>)
    where
        V: Clone,
    {
        // println!("remove_child w:{:x} n:{}", w, n);

        let old_child_ptr = match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { childs, .. } => childs[n].load(SeqCst),
            _ => unreachable!(),
        };

        let mut node = op.cas.alloc_node('t');
        node.trie_copy_from(unsafe { op.old.as_ref().unwrap() });

        let compact = match node.as_mut() {
            Node::Trie { childs, .. } => {
                childs.remove(n);
                childs.len() < 2
            }
            _ => unreachable!(),
        };
        node.hamming_reset(w);

        let new = Box::leak(node);

        op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
        op.cas.free_on_fail(gc::Mem::Node(new));
        op.cas.free_on_pass(gc::Mem::Node(op.old));
        if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
            (compact, CasRc::Ok(None))
        } else {
            (compact, CasRc::Retry)
        }
    }

    fn compact_from(
        &mut self,
        old_bmp: u16,
        old_childs: &[AtomicPtr<Child<V>>], // list to compact
        cas: &mut Cas<V>,
    ) where
        V: Clone,
    {
        let childs = match self {
            Node::Trie { bmp, childs } => {
                *bmp = old_bmp;
                childs
            }
            _ => unreachable!(),
        };

        childs.clear();

        for child in old_childs.iter() {
            let old_child_ptr = child.load(SeqCst);
            match unsafe { old_child_ptr.as_ref().unwrap() } {
                Child::Leaf(_) => childs.push(AtomicPtr::new(old_child_ptr)),
                Child::Deep(next_inode) => {
                    let next_node_ptr = next_inode.node.load(SeqCst);
                    match unsafe { next_node_ptr.as_ref().unwrap() } {
                        Node::Tomb { item } => {
                            cas.free_on_pass(gc::Mem::Node(next_node_ptr));
                            cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                            let new_ptr = Child::new_leaf(item.clone(), cas);
                            childs.push(AtomicPtr::new(new_ptr));
                        }
                        _ => childs.push(AtomicPtr::new(old_child_ptr)),
                    }
                }
            }
        }
    }
}

impl<V> Map<V> {
    pub fn get(&mut self, key: u32) -> Option<V>
    where
        V: Clone,
    {
        #[cfg(test)]
        let key = key_to_hash32(&key);

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

        let ws = slots(key);
        let res = 'retry: loop {
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            let mut wss = &ws[..];
            // print_ws!("get outer ws:{:?}", wss);

            loop {
                let old = inode.node.load(SeqCst);
                let node = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => break 'retry node.get_value(key),
                };
                wss = &wss[1..];
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
        #[cfg(test)]
        let key = key_to_hash32(&key);

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

        let ws = slots(key);
        let res = 'retry: loop {
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            let mut wss = &ws[..];
            // print_ws!("set try {:?}", ws);

            loop {
                let old: *mut Node<V> = inode.node.load(SeqCst);
                let node: &Node<V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => match node {
                        Node::Tomb { item } if item.key == key => continue 'retry,
                        Node::Tomb { .. } => break 'retry None,
                        Node::List { .. } => {
                            let op = generate_op!(self, inode, old);
                            match Node::update_list(key, &value, op) {
                                CasRc::Ok(old_value) => break 'retry old_value,
                                CasRc::Retry => continue 'retry,
                            }
                        }
                        Node::Trie { .. } => unreachable!(),
                    },
                };
                wss = &wss[1..];
                // println!("set loop w:{:x}", w);

                let n = match node {
                    Node::Trie { bmp, .. } => match hamming_distance(w, bmp.clone()) {
                        Distance::Insert(n) => {
                            // println!("set loop bmp:{:x} {}", bmp, n);
                            let op = generate_op!(self, inode, old);

                            let item = (key, value.clone()).into();
                            match Node::ins_child(item, w, n, op) {
                                CasRc::Ok(_) => break 'retry None,
                                CasRc::Retry => continue 'retry,
                            }
                        }
                        Distance::Set(n) => n,
                    },
                    Node::Tomb { item } if item.key == key => continue 'retry,
                    Node::Tomb { .. } => break 'retry None,
                    Node::List { .. } => unreachable!(),
                };
                // println!("set loop n:{}", n);

                inode = match unsafe { node.get_child(n).as_ref().unwrap() } {
                    Child::Deep(next_node) => next_node,
                    Child::Leaf(ot) if ot.key == key => {
                        let op = generate_op!(self, inode, old);

                        match Node::set_child((key, value.clone()).into(), n, op) {
                            CasRc::Ok(_) => {
                                break 'retry Some(ot.value.as_ref().cloned().unwrap());
                            }
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(_) if wss.len() == 0 => {
                        let op = generate_op!(self, inode, old);

                        match Node::leaf_to_list(key, &value, n, op) {
                            CasRc::Ok(_) => break 'retry None,
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(leaf) => {
                        let mut op = generate_op!(self, inode, old);

                        let mut scratch = [(0_u8, 0_u8); 8];
                        let xs = subtrie_zip(leaf.key, wss, &mut scratch);
                        // println!("set loop xs:{:?}", xs);

                        let item: Item<V> = (key, value.clone()).into();
                        let node_ptr = Node::new_subtrie(item, leaf, xs, &mut op);

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
        #[cfg(test)]
        let key = key_to_hash32(&key);

        let epoch = self.epoch.load(SeqCst);
        let (gc_epoch, compact, res) = {
            let (access_log, compact, res) = self.do_remove(key);
            (gc_epoch!(access_log, epoch), compact, res)
        };
        if compact {
            self.do_compact(key)
        }
        if gc_epoch < u64::MAX {
            self.cas.garbage_collect(gc_epoch)
        }
        res
    }

    fn do_remove(&mut self, key: u32) -> (RGuard, bool, Option<V>)
    where
        V: Clone,
    {
        let _epoch = self.generate_epoch();
        let access_log = self.access_log.read().expect("fail-lock");

        let ws = slots(key);
        let (compact, res) = 'retry: loop {
            let mut inode: &In<V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            let mut wss = &ws[..];
            // print_ws!("remove try {:?}", wss);

            loop {
                let old: *mut Node<V> = inode.node.load(SeqCst);
                let node: &Node<V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => match node {
                        Node::List { items } if items.len() > 0 => {
                            match items.iter().enumerate().find(|(_, x)| x.key == key) {
                                Some((n, _)) => {
                                    let op = generate_op!(self, inode, old);
                                    match Node::remove_from_list(n, op) {
                                        (_, CasRc::Retry) => continue 'retry,
                                        (comp, CasRc::Ok(ov)) => break 'retry (comp, ov),
                                    }
                                }
                                None => break 'retry (false, None),
                            }
                        }
                        Node::List { .. } => break 'retry (true, None), // empty list
                        Node::Tomb { item } if item.key == key => continue 'retry,
                        Node::Tomb { .. } => break 'retry (false, None),
                        Node::Trie { .. } => unreachable!(),
                    },
                };

                wss = &wss[1..];
                // println!("do_remove w:{:x}", w);

                let (n, _bmp, childs) = match node {
                    Node::Tomb { .. } => continue 'retry,
                    Node::Trie { bmp, childs } => match childs.len() {
                        0 => break 'retry (true, None),
                        _ => match hamming_distance(w, bmp.clone()) {
                            Distance::Insert(_) => break 'retry (false, None),
                            Distance::Set(n) => (n, bmp, childs),
                        },
                    },
                    Node::List { .. } => unreachable!(),
                };
                // println!("do_remove n:{} bmp:{:x}", n, _bmp);

                let old_child_ptr = childs[n].load(SeqCst);
                inode = match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Deep(next_inode) => next_inode,
                    Child::Leaf(item) if item.key == key => {
                        let op = generate_op!(self, inode, old);
                        let ov = item.value.as_ref().cloned().unwrap();
                        match Node::remove_child(w, n, op) {
                            (compact, CasRc::Ok(_)) => break 'retry (compact, Some(ov)),
                            (_, CasRc::Retry) => continue 'retry,
                        }
                    }
                    Child::Leaf(_) => break 'retry (false, None),
                }
            }
        };

        (access_log, compact, res)
    }

    fn do_compact(&mut self, key: u32)
    where
        V: Clone,
    {
        let _epoch = self.generate_epoch();
        let _access_log = self.access_log.read().expect("fail-lock");

        let ws = slots(key);
        'retry: loop {
            let mut inode: &In<V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
            let mut wss = &ws[..];
            // print_ws!("compact try {:?}", wss);

            let mut depth = 0;
            loop {
                depth += 1;
                let old: *mut Node<V> = inode.node.load(SeqCst);
                let node: &Node<V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => break 'retry,
                };
                wss = &wss[1..];
                // println!("do_compact w:{:x}", w);

                let n = match node {
                    Node::Tomb { .. } => continue 'retry,
                    Node::Trie { bmp, .. } => {
                        let hd = hamming_distance(w, bmp.clone());
                        match hd {
                            Distance::Insert(_) => break 'retry,
                            Distance::Set(n) => n,
                        }
                    }
                    Node::List { .. } => unreachable!(),
                };

                inode = match node {
                    Node::Trie { bmp, childs } if has_tomb_child(childs) && depth > 1 => {
                        let op = generate_op!(self, inode, old);

                        let mut node = op.cas.alloc_node('t');
                        node.compact_from(*bmp, childs, op.cas);
                        let new = Box::leak(node);

                        op.cas.free_on_fail(gc::Mem::Node(new));
                        op.cas.free_on_pass(gc::Mem::Node(old));
                        if op.cas.swing(op.epoch, &inode.node, old, new) {
                            match unsafe { childs[n].load(SeqCst).as_ref().unwrap() } {
                                Child::Leaf(_) => break 'retry,
                                Child::Deep(inode) => inode,
                            }
                        } else {
                            continue 'retry;
                        }
                    }
                    _ => break 'retry,
                }
            }
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
pub fn key_to_hash32<K>(key: &K) -> u32
where
    K: Hash + ?Sized,
{
    use fasthash::city::crc;

    let mut hasher = crc::Hasher128::default();
    key.hash(&mut hasher);
    let code: u64 = hasher.finish();
    (((code >> 32) ^ code) & 0xFFFFFFFF) as u32
}

fn slots(key: u32) -> [u8; 8] {
    let mut arr = [0_u8; 8];
    for i in 0..8 {
        arr[i] = ((key >> (i * 4)) & SLOT_MASK) as u8;
    }
    arr
}

fn subtrie_zip<'a>(key: u32, wss: &[u8], out: &'a mut [(u8, u8); 8]) -> &'a [(u8, u8)] {
    let ls = slots(key);
    let n = wss.len();
    let off = 8 - n;
    for i in 0..n {
        out[i] = (wss[i], ls[off + i])
    }
    &out[..n]
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

fn update_into_list<V>(item: Item<V>, items: &mut Vec<Item<V>>) -> Option<V>
where
    V: Clone,
{
    match items.iter().enumerate().find(|(_, x)| x.key == item.key) {
        Some((i, x)) => {
            let old_value = x.value.clone();
            items[i] = item;
            Some(old_value.unwrap())
        }
        None => {
            items.push(item);
            None
        }
    }
}

fn has_tomb_child<V>(childs: &[AtomicPtr<Child<V>>]) -> bool {
    childs
        .iter()
        .any(|child| unsafe { child.load(SeqCst).as_ref().unwrap() }.is_tomb_node())
}

#[cfg(test)]
#[path = "map_test.rs"]
mod map_test;
