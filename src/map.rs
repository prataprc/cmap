use std::{
    borrow::Borrow,
    fmt::Debug,
    hash::{Hash, Hasher},
    mem,
    ops::{Add, Deref},
    sync::{
        self,
        atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst},
        Arc, RwLock,
    },
    thread, time,
};

use crate::gc::{self, Cas, Epoch};

const SLOT_MASK: u32 = 0xF;

#[allow(unused_macros)]
macro_rules! format_ws {
    ($fmt:expr, $ws:expr) => {{
        let ws = $ws
            .iter()
            .map(|w| format!("{:x}", w))
            .collect::<Vec<String>>();
        format!($fmt, ws)
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

// TODO: atomic-ordering replace SeqCst with Acquire/Release.
// TODO: Rename set() API to insert() API.
// TODO: stats() method
//   * Count the number of Nodes that are tombable.
// TODO: compact() logic
//   * Compact trie-childs to capacity == length.
//   * Compact list-items to capacity == length.
// TODO: n_compacts and n_retries accounting in test/dev mode.

type RGuard<'a> = sync::RwLockReadGuard<'a, Vec<Arc<AtomicU64>>>;

pub struct Map<K, V> {
    id: usize,
    root: Arc<Root<K, V>>,

    epoch: Arc<AtomicU64>,
    access_log: Arc<RwLock<Vec<Arc<AtomicU64>>>>,
    cas: gc::Cas<K, V>,
    n_pools: Arc<AtomicUsize>,
    n_allocs: Arc<AtomicUsize>,
    n_frees: Arc<AtomicUsize>,
}

pub struct In<K, V> {
    node: AtomicPtr<Node<K, V>>,
}

pub enum Node<K, V> {
    Trie {
        bmp: u16,
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

#[derive(Clone, Default, PartialEq, Debug)]
pub struct Item<K, V> {
    key: K,
    value: V,
}

impl<K, V> Default for Child<K, V>
where
    K: Default,
    V: Default,
{
    fn default() -> Self {
        Child::Leaf(Item::default())
    }
}

pub struct Root<K, V> {
    root: AtomicPtr<In<K, V>>,
}

impl<K, V> Deref for Root<K, V> {
    type Target = AtomicPtr<In<K, V>>;

    fn deref(&self) -> &AtomicPtr<In<K, V>> {
        &self.root
    }
}

impl<K, V> Drop for Root<K, V> {
    fn drop(&mut self) {
        let inode = unsafe { Box::from_raw(self.root.load(SeqCst)) };
        Node::dropped(inode.node.load(SeqCst));
    }
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
    K: Default + Clone + Debug,
    V: Default + Clone + Debug,
{
    fn print(&self, prefix: &str) {
        let node = unsafe { self.node.load(SeqCst).as_ref().unwrap() };
        node.print(prefix)
    }

    fn validate(&self, depth: usize) -> Stats {
        unsafe { self.node.load(SeqCst).as_ref().unwrap().validate(depth + 1) }
    }

    #[cfg(test)]
    fn collisions(&self)
    where
        K: Default + Clone + Hash,
    {
        unsafe { self.node.load(SeqCst).as_ref().unwrap().collisions() };
    }
}

impl<K, V> Drop for Map<K, V> {
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
        self.n_pools.fetch_add(self.cas.to_pools_len(), SeqCst);
        self.n_allocs.fetch_add(self.cas.to_alloc_count(), SeqCst);
        self.n_frees.fetch_add(self.cas.to_free_count(), SeqCst);
    }
}

impl<K, V> Map<K, V>
where
    K: 'static + Send + Default + Clone,
    V: 'static + Send + Default + Clone,
{
    pub fn new() -> Map<K, V> {
        let mut cas = gc::Cas::new();
        let root = {
            let node = cas.alloc_node('t');
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
            cas,
            n_pools: Arc::new(AtomicUsize::new(0)),
            n_allocs: Arc::new(AtomicUsize::new(0)),
            n_frees: Arc::new(AtomicUsize::new(0)),
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
            cas: gc::Cas::new(),
            n_pools: Arc::clone(&self.n_pools),
            n_allocs: Arc::clone(&self.n_allocs),
            n_frees: Arc::clone(&self.n_frees),
        }
    }

    pub fn validate(&self) -> Stats
    where
        K: Debug,
        V: Debug,
    {
        let _epoch = self.epoch.load(SeqCst);
        let _guard = self.access_log.write().expect("lock-panic");

        let mut stats = unsafe { self.root.load(SeqCst).as_ref().unwrap().validate(0) };
        stats.n_pools = self.n_pools.load(SeqCst) + self.cas.to_pools_len();
        stats.n_allocs = self.n_allocs.load(SeqCst) + self.cas.to_alloc_count();
        stats.n_frees = self.n_frees.load(SeqCst) + self.cas.to_free_count();

        self.cas.validate();

        {
            let mem_count = stats.n_allocs - stats.n_frees;
            let alg_count = stats.n_nodes + stats.n_childs + stats.n_pools;
            debug_assert!(
                mem_count == alg_count,
                "mem_count:{} alg_count:{}",
                mem_count,
                alg_count
            );
        }

        stats
    }

    #[cfg(test)]
    pub fn collisions(&self)
    where
        K: Hash + Debug,
        V: Debug,
    {
        let _epoch = self.epoch.load(SeqCst);
        let _guard = self.access_log.write().expect("lock-panic");

        unsafe { self.root.load(SeqCst).as_ref().unwrap().collisions() };
    }
}

impl<K, V> Map<K, V>
where
    K: Default + Clone + Debug,
    V: Default + Clone + Debug,
{
    pub fn print(&self)
    where
        K: Debug,
        V: Debug,
    {
        let epoch = self.epoch.load(SeqCst);
        let guard = self.access_log.write().expect("lock-panic");
        let access_log = guard.iter().map(|e| e.load(SeqCst));
        println!("Map<{},{:?}>", epoch, access_log);

        unsafe { self.root.load(SeqCst).as_ref().unwrap().print("  ") };
    }

    pub fn print_sizing(&self) {
        println!("size of node {:4}", mem::size_of::<Node<K, V>>());
        println!(
            "size of aptr {:4}",
            mem::size_of::<AtomicPtr<Child<K, V>>>()
        );
        println!("size of chil {:4}", mem::size_of::<Child<K, V>>());
        println!("size of item {:4}", mem::size_of::<Item<K, V>>());
        println!("size of inod {:4}", mem::size_of::<In<K, V>>());
    }
}

impl<K, V> Map<K, V> {
    fn generate_epoch(&self, access_log: &RGuard) -> Epoch {
        let epoch = {
            let at = Arc::clone(&access_log[self.id]);
            let epoch = Arc::clone(&self.epoch);
            Epoch::new(epoch, at)
        };

        epoch
    }
}

impl<K, V> Child<K, V>
where
    K: Default + Clone,
    V: Default + Clone,
{
    fn new_leaf(leaf: Item<K, V>, cas: &mut gc::Cas<K, V>) -> *mut Child<K, V> {
        let mut child = cas.alloc_child();

        *child = Child::Leaf(leaf.clone());

        let child_ptr = Box::leak(child);
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }

    fn new_deep(node: *mut Node<K, V>, cas: &mut gc::Cas<K, V>) -> *mut Child<K, V> {
        let inode = In {
            node: AtomicPtr::new(node),
        };

        let mut child = cas.alloc_child();

        *child = Child::Deep(inode);

        let child_ptr = Box::leak(child);
        cas.free_on_fail(gc::Mem::Child(child_ptr));
        child_ptr
    }

    fn is_leaf(&self) -> bool {
        match self {
            Child::Leaf(_) => true,
            Child::Deep(_) => false,
        }
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

    fn is_empty_list_node(&self) -> bool {
        match self {
            Child::Leaf(_) => false,
            Child::Deep(inode) => {
                let node = unsafe { inode.node.load(SeqCst).as_ref().unwrap() };
                match node {
                    Node::List { items } if items.len() == 0 => true,
                    _ => false,
                }
            }
        }
    }
}

impl<K, V> Child<K, V> {
    fn dropped(child: *mut Child<K, V>) {
        let child = unsafe { Box::from_raw(child) };
        match child.as_ref() {
            Child::Leaf(_item) => (),
            Child::Deep(inode) => Node::dropped(inode.node.load(SeqCst)),
        }
    }
}

impl<K, V> Node<K, V>
where
    K: Default + Clone,
    V: Default + Clone,
{
    #[inline]
    fn get_child(&self, n: usize) -> *mut Child<K, V> {
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

    fn get_value<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: PartialEq + Hash + ?Sized,
        V: Clone,
    {
        match self {
            Node::List { items } => get_from_list(key, items),
            Node::Tomb { item } if item.key.borrow() == key => Some(item.value.clone()),
            Node::Tomb { .. } => None,
            Node::Trie { .. } => None,
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
}

impl<K, V> Node<K, V> {
    fn dropped(node: *mut Node<K, V>) {
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

impl<K, V> Node<K, V>
where
    K: Default + Clone + Debug,
    V: Default + Clone + Debug,
{
    fn print(&self, prefix: &str) {
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

    // make sure that Node::List are only at the last level.
    fn validate(&self, depth: usize) -> Stats {
        let mut stats = Stats::default();
        stats.n_nodes += 1;

        stats.n_mem += mem::size_of::<Node<K, V>>();
        match self {
            Node::Trie { childs, .. } => {
                debug_assert!(childs.len() <= 16);
                stats.n_childs += childs.len();

                stats.n_mem += {
                    let n = mem::size_of::<AtomicPtr<Child<K, V>>>();
                    let m = mem::size_of::<Child<K, V>>();
                    (childs.capacity() * n) + (childs.len() * m)
                };
                for child in childs {
                    match unsafe { child.load(SeqCst).as_ref().unwrap() } {
                        Child::Leaf(_) => stats.n_items += 1,
                        Child::Deep(inode) => stats = stats + inode.validate(depth),
                    }
                }
            }
            Node::Tomb { .. } => {
                stats.n_tombs += 1;
                stats.n_items += 1;
            }
            Node::List { items } => {
                debug_assert!(depth == 9, "depth:{}", depth);
                stats.n_lists += 1;
                stats.n_items += items.len();
                stats.n_mem += items.capacity() * mem::size_of::<Item<K, V>>();
            }
        }

        stats
    }

    #[cfg(test)]
    fn collisions(&self)
    where
        K: Hash,
    {
        match self {
            Node::Trie { childs, .. } => {
                for child in childs {
                    match unsafe { child.load(SeqCst).as_ref().unwrap() } {
                        Child::Leaf(_) => (),
                        Child::Deep(inode) => inode.collisions(),
                    }
                }
            }
            Node::Tomb { .. } => (),
            Node::List { items } => {
                for item in items {
                    println!("key:{:?},{}", item.key, key_to_hash32(&item.key))
                }
            }
        }
    }
}

impl<K, V> Node<K, V>
where
    K: Default + Clone,
    V: Default + Clone,
{
    fn new_bi_list(item: Item<K, V>, leaf: &Item<K, V>, cas: &mut Cas<K, V>) -> *mut Node<K, V> {
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

    fn new_list_with(
        olds: &[Item<K, V>],
        item: Item<K, V>,
        cas: &mut Cas<K, V>,
    ) -> (*mut Node<K, V>, Option<V>)
    where
        K: PartialEq,
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

    fn new_list_without(olds: &[Item<K, V>], i: usize, cas: &mut Cas<K, V>) -> *mut Node<K, V> {
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

    fn new_tomb(nitem: &Item<K, V>, cas: &mut Cas<K, V>) -> *mut Node<K, V> {
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
        item: Item<K, V>,
        leaf: &Item<K, V>,
        pairs: &[(u8, u8)],
        op: &mut CasOp<K, V>,
    ) -> *mut Node<K, V> {
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

    fn trie_copy_from(&mut self, node: &Node<K, V>) {
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

    fn update_list(key: &K, value: &V, op: CasOp<K, V>) -> CasRc<Option<V>>
    where
        K: PartialEq,
    {
        match unsafe { op.old.as_ref().unwrap() } {
            Node::List { items: olds } => {
                let item: Item<K, V> = (key.clone(), value.clone()).into();
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

    fn ins_child(leaf: Item<K, V>, w: u8, n: usize, op: CasOp<K, V>) -> CasRc<()> {
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

    fn set_child(leaf: Item<K, V>, n: usize, op: CasOp<K, V>) -> CasRc<()> {
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

    fn leaf_to_list(key: K, value: &V, n: usize, op: CasOp<K, V>) -> CasRc<()> {
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
                        let item: Item<K, V> = (key, value.clone()).into();
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

    fn set_trie_child(node: *mut Node<K, V>, n: usize, op: CasOp<K, V>) -> CasRc<()> {
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

    fn remove_from_list(n: usize, op: CasOp<K, V>) -> (bool, CasRc<Option<V>>) {
        let (compact, new, ov) = match unsafe { op.old.as_ref().unwrap() } {
            Node::List { items } if items.len() == 1 => (
                true,
                Node::new_list_without(items, n, op.cas),
                items[n].value.clone(),
            ),
            Node::List { items } if items.len() == 2 => {
                let j = [1, 0][n];
                (
                    true,
                    Node::new_tomb(&items[j], op.cas),
                    items[n].value.clone(),
                )
            }
            Node::List { items } if items.len() > 0 => (
                false,
                Node::new_list_without(items, n, op.cas),
                items[n].value.clone(),
            ),
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

    fn remove_child(w: u8, n: usize, op: CasOp<K, V>) -> (bool, CasRc<Option<V>>) {
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
                has_single_child_leaf(childs)
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

    fn compact_trie_from(
        &mut self,
        old_bmp: u16,
        old_childs: &[AtomicPtr<Child<K, V>>], // list to compact
        cas: &mut Cas<K, V>,
    ) -> bool {
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
                        Node::List { items } if items.len() == 0 => {
                            cas.free_on_pass(gc::Mem::Node(next_node_ptr));
                            cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                            // skip
                        }
                        _ => childs.push(AtomicPtr::new(old_child_ptr)),
                    }
                }
            }
        }

        has_single_child_leaf(childs)
    }

    fn compact_tomb_from(
        &mut self,
        old_childs: &[AtomicPtr<Child<K, V>>], // list to compact
        cas: &mut Cas<K, V>,
    ) {
        debug_assert!(old_childs.len() == 1);

        match self {
            Node::Tomb { item } => {
                let old_child_ptr = old_childs[0].load(SeqCst);
                match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Leaf(leaf) => {
                        cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                        item.clone_from(leaf);
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }
}

impl<K, V> Map<K, V>
where
    K: Default + Clone,
    V: Default + Clone,
{
    pub fn get<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: PartialEq + Hash + ?Sized,
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

    fn do_get<Q>(&mut self, key: &Q) -> (RGuard, Option<V>)
    where
        K: Borrow<Q>,
        Q: PartialEq + Hash + ?Sized,
    {
        let access_log = self.access_log.read().expect("fail-lock");
        let _epocher = self.generate_epoch(&access_log);

        let ws = slots(key_to_hash32(key));
        let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
        let mut wss = &ws[..];
        // format_ws!("get outer ws:{:?}", wss);

        let res = loop {
            let old = inode.node.load(SeqCst);
            let node = unsafe { old.as_ref().unwrap() };

            let w = match wss.first() {
                Some(w) => *w,
                None => break node.get_value(key),
            };
            wss = &wss[1..];
            // println!("get loop w:{:x}", w);

            inode = match node {
                Node::Trie { bmp, childs } => {
                    let hd = hamming_distance(w, bmp.clone());
                    // println!("get loop bmp:{:x} {:?}", bmp, hd);
                    match hd {
                        Distance::Insert(_) => break None,
                        Distance::Set(n) => {
                            let ptr = childs[n].load(SeqCst);
                            match unsafe { ptr.as_ref().unwrap() } {
                                Child::Deep(next_inode) => next_inode,
                                Child::Leaf(item) if item.key.borrow() == key => {
                                    break Some(item.value.clone());
                                }
                                Child::Leaf(_) => break None,
                            }
                        }
                    }
                }
                Node::List { .. } => unreachable!(),
                Node::Tomb { item } if item.key.borrow() == key => {
                    break Some(item.value.clone());
                }
                Node::Tomb { .. } => break None,
            }
        };
        (access_log, res)
    }

    pub fn set(&mut self, key: K, value: V) -> Option<V>
    where
        K: PartialEq + Hash,
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

    fn do_set(&mut self, key: K, value: V) -> (RGuard, Option<V>)
    where
        K: PartialEq + Hash,
    {
        let access_log = self.access_log.read().expect("fail-lock");
        let _epocher = self.generate_epoch(&access_log);

        let ws = slots(key_to_hash32(&key));
        let res = 'retry: loop {
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            let mut wss = &ws[..];
            // println!("set try key:{:?} {:?}", key, format_ws!("{:?}", ws));

            loop {
                let old: *mut Node<K, V> = inode.node.load(SeqCst);
                let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => match node {
                        Node::Tomb { item } if &item.key == &key => {
                            // println!("set loop tip tomb equal {:?}", key);
                            continue 'retry;
                        }
                        Node::Tomb { .. } => {
                            // println!("set loop tip tomb {:?}", key);
                            break 'retry None;
                        }
                        Node::List { .. } => {
                            let op = generate_op!(self, inode, old);
                            match Node::update_list(&key, &value, op) {
                                CasRc::Ok(old_value) => break 'retry old_value,
                                CasRc::Retry => continue 'retry,
                            }
                        }
                        Node::Trie { .. } => unreachable!(),
                    },
                };
                wss = &wss[1..];
                // println!("set loop w:{:x} wss:{:?}", w, wss);

                let n = match node {
                    Node::Trie { bmp, .. } => match hamming_distance(w, bmp.clone()) {
                        Distance::Insert(n) => {
                            // println!("set loop bmp:{:x} {}", bmp, n);
                            let op = generate_op!(self, inode, old);

                            let item = (key.clone(), value.clone()).into();
                            match Node::ins_child(item, w, n, op) {
                                CasRc::Ok(_) => break 'retry None,
                                CasRc::Retry => continue 'retry,
                            }
                        }
                        Distance::Set(n) => n,
                    },
                    Node::Tomb { item } if &item.key == &key => continue 'retry,
                    Node::Tomb { .. } => break 'retry None,
                    Node::List { .. } => unreachable!(),
                };
                // println!("set loop n:{}", n);

                inode = match unsafe { node.get_child(n).as_ref().unwrap() } {
                    Child::Deep(next_node) => {
                        // println!("set loop next level {:?}", key);
                        next_node
                    }
                    Child::Leaf(ot) if &ot.key == &key => {
                        let op = generate_op!(self, inode, old);
                        // println!("set loop 1");

                        let item = (key.clone(), value.clone()).into();
                        match Node::set_child(item, n, op) {
                            CasRc::Ok(_) => break 'retry Some(ot.value.clone()),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(_) if wss.len() == 0 => {
                        let op = generate_op!(self, inode, old);
                        // println!("set loop 2");

                        match Node::leaf_to_list(key.clone(), &value, n, op) {
                            CasRc::Ok(_) => break 'retry None,
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(leaf) => {
                        let mut op = generate_op!(self, inode, old);

                        let mut scratch = [(0_u8, 0_u8); 8];
                        let xs = subtrie_zip(&leaf.key, wss, &mut scratch);
                        // println!("set loop 3");

                        let item: Item<K, V> = (key.clone(), value.clone()).into();
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

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: PartialEq + Hash + ?Sized,
    {
        let epoch = self.epoch.load(SeqCst);
        let (gc_epoch, mut compact, res) = {
            let (access_log, compact, res) = self.do_remove(key);
            (gc_epoch!(access_log, epoch), compact, res)
        };
        while compact {
            compact = self.do_compact(key)
        }
        if gc_epoch < u64::MAX {
            self.cas.garbage_collect(gc_epoch)
        }
        res
    }

    fn do_remove<Q>(&mut self, key: &Q) -> (RGuard, bool, Option<V>)
    where
        K: Borrow<Q>,
        Q: PartialEq + Hash + ?Sized,
    {
        let access_log = self.access_log.read().expect("fail-lock");
        let _epocher = self.generate_epoch(&access_log);

        let ws = slots(key_to_hash32(key));
        let (compact, res) = 'retry: loop {
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };

            let mut wss = &ws[..];
            // println!("remove try key:{:?} {:?}", key, format_ws!("{:?}", ws));

            loop {
                let old: *mut Node<K, V> = inode.node.load(SeqCst);
                let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => match node {
                        Node::List { items } if items.len() > 0 => {
                            match items
                                .iter()
                                .enumerate()
                                .find(|(_, x)| x.key.borrow() == key)
                            {
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
                        Node::Tomb { item } if item.key.borrow() == key => {
                            continue 'retry;
                        }
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
                    Child::Leaf(item) if item.key.borrow() == key => {
                        let op = generate_op!(self, inode, old);
                        let ov = item.value.clone();
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

    fn do_compact<Q>(&mut self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        let _access_log = self.access_log.read().expect("fail-lock");
        let _epocher = self.generate_epoch(&_access_log);

        let ws = slots(key_to_hash32(key));
        'retry: loop {
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
            let mut wss = &ws[..];
            //format_ws!("do_compact try {:?}", wss);

            let mut depth = 0;
            loop {
                depth += 1;
                let old: *mut Node<K, V> = inode.node.load(SeqCst);
                let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => match node {
                        Node::List { items } if items.len() < 2 => break 'retry true,
                        Node::Tomb { .. } => break 'retry true,
                        Node::List { .. } => break 'retry false,
                        Node::Trie { .. } => unreachable!(),
                    },
                };
                wss = &wss[1..];
                // println!("do_compact w:{:x}", w);

                let n = match node {
                    Node::Tomb { .. } => continue 'retry,
                    Node::Trie { bmp, .. } => {
                        let hd = hamming_distance(w, bmp.clone());
                        match hd {
                            Distance::Insert(_) => break 'retry false,
                            Distance::Set(n) => n,
                        }
                    }
                    Node::List { .. } => unreachable!(),
                };

                inode = match node {
                    Node::Trie { childs, .. } if depth == 1 => {
                        match unsafe { childs[n].load(SeqCst).as_ref().unwrap() } {
                            Child::Leaf(_) => break 'retry false,
                            Child::Deep(inode) => inode,
                        }
                    }
                    Node::Trie { childs, .. } if has_single_child_leaf(childs) => {
                        let op = generate_op!(self, inode, old);

                        let mut node = op.cas.alloc_node('b');
                        node.compact_tomb_from(childs, op.cas);
                        let new = Box::leak(node);

                        op.cas.free_on_fail(gc::Mem::Node(new));
                        op.cas.free_on_pass(gc::Mem::Node(old));
                        if op.cas.swing(op.epoch, &inode.node, old, new) {
                            break 'retry true;
                        } else {
                            continue 'retry;
                        }
                    }
                    Node::Trie { bmp, childs } if has_tomb_empty_child(childs) => {
                        let op = generate_op!(self, inode, old);

                        let mut node = op.cas.alloc_node('t');
                        let compact = node.compact_trie_from(*bmp, childs, op.cas);
                        let new = Box::leak(node);

                        op.cas.free_on_fail(gc::Mem::Node(new));
                        op.cas.free_on_pass(gc::Mem::Node(old));
                        if op.cas.swing(op.epoch, &inode.node, old, new) {
                            if compact {
                                break 'retry true;
                            }
                            match unsafe { childs[n].load(SeqCst).as_ref().unwrap() } {
                                Child::Leaf(_) => break 'retry false,
                                Child::Deep(inode) => inode,
                            }
                        } else {
                            continue 'retry;
                        }
                    }
                    Node::Trie { childs, .. } => {
                        match unsafe { childs[n].load(SeqCst).as_ref().unwrap() } {
                            Child::Leaf(_) => break 'retry false,
                            Child::Deep(inode) => inode,
                        }
                    }
                    _ => break 'retry false,
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        let _access_log = self.access_log.write().expect("fail-lock");

        let inode: &In<K, V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
        let node = unsafe { inode.node.load(SeqCst).as_ref().unwrap() };
        node.count()
    }
}

enum CasRc<T> {
    Ok(T),
    Retry,
}

struct CasOp<'a, K, V> {
    epoch: &'a Arc<AtomicU64>,
    inode: &'a In<K, V>,
    old: *mut Node<K, V>,
    cas: &'a mut Cas<K, V>,
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
fn key_to_hash32<K>(key: &K) -> u32
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

fn subtrie_zip<'a, K>(key: &K, wss: &[u8], out: &'a mut [(u8, u8); 8]) -> &'a [(u8, u8)]
where
    K: Hash + ?Sized,
{
    let ls = slots(key_to_hash32(key));
    let n = wss.len();
    let off = 8 - n;
    for i in 0..n {
        out[i] = (wss[i], ls[off + i])
    }
    &out[..n]
}

fn get_from_list<K, V, Q>(key: &Q, items: &[Item<K, V>]) -> Option<V>
where
    K: Borrow<Q>,
    V: Clone,
    Q: PartialEq + Hash + ?Sized,
{
    match items.iter().find(|x| x.key.borrow() == key) {
        Some(x) => Some(x.value.clone()),
        None => None,
    }
}

fn update_into_list<K, V>(item: Item<K, V>, items: &mut Vec<Item<K, V>>) -> Option<V>
where
    K: PartialEq + Default + Clone,
    V: Default + Clone,
{
    match items.iter().enumerate().find(|(_, x)| x.key == item.key) {
        Some((i, x)) => {
            let old_value = x.value.clone();
            items[i] = item;
            Some(old_value)
        }
        None => {
            items.push(item);
            None
        }
    }
}

fn has_tomb_empty_child<K, V>(childs: &[AtomicPtr<Child<K, V>>]) -> bool
where
    K: Default + Clone,
    V: Default + Clone,
{
    childs.iter().any(|child| {
        let c = unsafe { child.load(SeqCst).as_ref().unwrap() };
        c.is_tomb_node() || c.is_empty_list_node()
    })
}

fn has_single_child_leaf<K, V>(childs: &[AtomicPtr<Child<K, V>>]) -> bool
where
    K: Default + Clone,
    V: Default + Clone,
{
    childs.len() == 1
        && childs
            .iter()
            .all(|child| unsafe { child.load(SeqCst).as_ref().unwrap() }.is_leaf())
}

// TODO: count the number of leaf nodes.
// TODO: count the sum total of leaf nodes depth, compute average.
#[derive(Default, Debug)]
pub struct Stats {
    pub n_nodes: usize,
    pub n_childs: usize,
    pub n_items: usize,
    pub n_tombs: usize,
    pub n_lists: usize,
    pub n_pools: usize,
    pub n_allocs: usize,
    pub n_frees: usize,
    pub n_mem: usize,
}

impl Add for Stats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Stats {
            n_nodes: self.n_nodes + rhs.n_nodes,
            n_childs: self.n_childs + rhs.n_childs,
            n_items: self.n_items + rhs.n_items,
            n_tombs: self.n_tombs + rhs.n_tombs,
            n_lists: self.n_lists + rhs.n_lists,
            n_pools: self.n_pools + rhs.n_pools,
            n_allocs: self.n_allocs + rhs.n_allocs,
            n_frees: self.n_frees + rhs.n_frees,
            n_mem: self.n_mem + rhs.n_mem,
        }
    }
}

#[cfg(test)]
#[path = "arr_test.rs"]
mod arr_test;
#[cfg(test)]
#[path = "dash_test.rs"]
mod dash_test;
#[cfg(test)]
#[path = "map_test.rs"]
mod map_test;
