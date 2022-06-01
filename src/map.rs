//! Module implement concurrent map.
//!
//! Following is the way the data structure is wired up.
//!
//! ```ignore
//! Map -> Root -> In -> Node --> Trie [ -> Child ] ---> Deep
//!                           |                     |--> Leaf
//!                           |                     |--> None
//!                           |-> Tomp { Item }
//!                           |-> List [ Item ]
//! ```
//!
//! We are using a combination of copy-on-write and atomic-ops, like load, store,
//! compare-exchange to support concurrent read/write access into the map.
//!
//! **Drop semantics**
//!
//! Drop is implemented by the [Map] type and its `Root` member. At the [Map] level,
//! before dropping the root and its trie, following cleanup is performed
//!
//! * Map::access_log is set to ZERO.
//! * Pending reclaims from the `gc::Cas` instance, which is not shared
//!   between Map's clone, are garbage collected, Refer to [gc] module for details
//!   on how epochal garbage collection works.
//! * stats like `n_pool`, `n_allocs` and `n_frees` are updated from the current clone's
//!   `cas` instance.
//!
//! Actual drop is executed by Node::dropped() function

use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::{Add, Deref};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::{borrow::Borrow, fmt::Debug, mem, thread, time};

use crate::{gc, DefaultHasher};

const SLOT_MASK: u32 = 0xF;
const ENTER_MASK: u64 = 0x8000000000000000;
const EPOCH_MASK: u64 = 0x7FFFFFFFFFFFFFFF;
const GC_PERIOD: usize = 16;

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
    ($log:expr, $seqno:expr) => {{
        let mut gc_epoch = u64::MAX;
        for e in $log.iter() {
            let thread_epoch = e.load(SeqCst);
            let thread_epoch = if thread_epoch == 0 {
                // map's clone is dropped
                gc_epoch
            } else if thread_epoch & ENTER_MASK == 0 {
                // thread exited at thread_epoch
                $seqno
            } else {
                // ongoing access.
                thread_epoch & EPOCH_MASK
            };
            gc_epoch = u64::min(gc_epoch, thread_epoch);
        }
        // gc_epoch is
        // a. either u64::MAX or,
        // b. epoch at an ongoing access,
        // c. current epoch, if there is no on-going access.
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

/// Map implement concurrent hash-map of key ``K`` and value ``V``.
pub struct Map<K, V, H = DefaultHasher> {
    id: usize,
    hash_builder: H,
    root: Arc<Root<K, V>>,

    epoch: Arc<AtomicU64>,
    access_log: Arc<Vec<AtomicU64>>,
    map_pool: Arc<Mutex<Vec<Map<K, V, H>>>>,
    cas: gc::Cas<K, V>,

    gc_period: usize,
    gc_count: usize,
    n_pools: Arc<AtomicUsize>,
    n_allocs: Arc<AtomicUsize>,
    n_frees: Arc<AtomicUsize>,
}

pub(crate) struct Root<K, V> {
    root: AtomicPtr<In<K, V>>,
}

pub(crate) struct In<K, V> {
    node: AtomicPtr<Node<K, V>>,
}

pub(crate) enum Node<K, V> {
    Trie {
        bmp: u16,
        childs: Vec<AtomicPtr<Child<K, V>>>,
    },
    Tomb {
        item: Option<Item<K, V>>,
    },
    List {
        items: Vec<Item<K, V>>,
    },
}

pub(crate) enum Child<K, V> {
    Deep(In<K, V>),
    Leaf(Item<K, V>),
    None,
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Item<K, V> {
    key: K,
    value: V,
}

impl<K, V> Default for Child<K, V> {
    fn default() -> Self {
        Child::None
    }
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
    #[cfg(any(test, feature = "perf"))]
    fn print(&self, prefix: &str) {
        println!("{}Item<{:?},{:?}>", prefix, self.key, self.value);
    }
}

impl<K, V> In<K, V>
where
    K: Clone,
    V: Clone,
{
    #[cfg(any(test, feature = "perf"))]
    fn print(&self, prefix: &str)
    where
        K: Debug,
        V: Debug,
    {
        let node = unsafe { self.node.load(SeqCst).as_ref().unwrap() };
        node.print(prefix)
    }

    fn validate(&self, depth: usize) -> Stats {
        unsafe { self.node.load(SeqCst).as_ref().unwrap().validate(depth + 1) }
    }

    #[cfg(test)]
    fn collisions<H>(&self, hb: &H)
    where
        K: Clone + Hash + Debug,
        H: BuildHasher,
    {
        unsafe { self.node.load(SeqCst).as_ref().unwrap().collisions(hb) };
    }
}

impl<K, V, H> Drop for Map<K, V, H> {
    fn drop(&mut self) {
        self.access_log[self.id].store(0, SeqCst);

        while self.cas.has_reclaims() {
            let seqno = self.epoch.load(SeqCst);
            let seqno = gc_epoch!(self.access_log, seqno);
            if seqno == u64::MAX {
                // force collect, either all clones have been dropped or there was none.
                self.cas.garbage_collect(u64::MAX)
            } else {
                self.cas.garbage_collect(seqno)
            }
            thread::sleep(time::Duration::from_millis(10));
        }
        self.n_pools.fetch_add(self.cas.to_pools_len(), SeqCst);
        self.n_allocs.fetch_add(self.cas.to_alloc_count(), SeqCst);
        self.n_frees.fetch_add(self.cas.to_free_count(), SeqCst);
    }
}

impl<K, V, H> Map<K, V, H> {
    /// Create a new instance of map. All the clones created from this map will
    /// share its internal data structure through atomic serialization.
    ///
    /// If application intent to use this map in single threaded mode, supply
    /// `concurrency` as 1. Otherwise supplied level of concurrency must be equal
    /// to or greater than the number of times this intance is going to be cloned.
    pub fn new(concurrency: usize, hash_builder: H) -> Map<K, V, H>
    where
        H: Clone,
    {
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

        let mut access_log = vec![];
        (0..concurrency).for_each(|_| access_log.push(AtomicU64::new(1)));

        let map = Map {
            id: 0,
            hash_builder,
            root,

            epoch: Arc::new(AtomicU64::new(1)),
            access_log: Arc::new(access_log),
            map_pool: Arc::new(Mutex::new(vec![])),
            cas,
            gc_period: GC_PERIOD,
            gc_count: GC_PERIOD,
            n_pools: Arc::new(AtomicUsize::new(0)),
            n_allocs: Arc::new(AtomicUsize::new(0)),
            n_frees: Arc::new(AtomicUsize::new(0)),
        };

        map.create_map_pool((1..concurrency).collect());

        map
    }

    /// Get a pre-created a clone of Map instance. Cloned instances are thread-safe
    /// and can be use across threads to share the same underlying map.
    pub fn cloned(&self) -> Map<K, V, H> {
        self.map_pool
            .lock()
            .expect("map lock poisoned")
            .pop()
            .unwrap()
    }

    fn create_map_pool(&self, ids: Vec<usize>)
    where
        H: Clone,
    {
        for id in ids.into_iter() {
            let map = Map {
                id,
                hash_builder: self.hash_builder.clone(),
                root: Arc::clone(&self.root),

                epoch: Arc::clone(&self.epoch),
                access_log: Arc::clone(&self.access_log),
                map_pool: Arc::clone(&self.map_pool),
                cas: gc::Cas::new(),
                gc_period: self.gc_period,
                gc_count: self.gc_count,
                n_pools: Arc::clone(&self.n_pools),
                n_allocs: Arc::clone(&self.n_allocs),
                n_frees: Arc::clone(&self.n_frees),
            };
            self.map_pool.lock().expect("map lock poisoned").push(map);
        }
    }

    /// Data structure internally uses epoch based garbage collection for safety
    /// and optimization. Garbage collection is per map-clone/thread and for
    /// each thread gc will be triggered for every mutation operation. This can
    /// be costly since this require accessing an array of atomically protected
    /// 64-bit seqno. By setting this to N, gc will be triggered,
    /// for each thread, for every N mutations.
    pub fn set_gc_period(&mut self, period: usize) -> &mut Self {
        self.gc_period = period;
        self
    }

    /// Return Map's clone id.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Return current epoch across all threads/clones
    pub fn epoch(&self) -> u64 {
        self.epoch.load(SeqCst)
    }

    /// Return garbage collectible epoch. epoch is either u64::MAX, in which case all the
    /// clones of this Map is dropped, or the safe minimum epoch where all allocations
    /// that happed before epoch can be safely gargage collected
    pub fn gc_epoch(&self, seqno: u64) -> u64 {
        let mut gc_epoch = u64::MAX;
        for e in self.access_log.iter() {
            let thread_epoch = e.load(SeqCst);
            let thread_epoch = if thread_epoch == 0 {
                // map's clone is dropped
                gc_epoch
            } else if thread_epoch & ENTER_MASK == 0 {
                // thread exited at thread_epoch
                seqno
            } else {
                // ongoing access.
                thread_epoch & EPOCH_MASK
            };
            gc_epoch = u64::min(gc_epoch, thread_epoch);
        }
        gc_epoch
    }

    /// Return the number of items indexed in the map. This may not be accurate due
    /// to concurrent writes. Note that this is a costly operation walking through
    /// the entire map.
    pub fn len(&self) -> usize {
        let inode: &In<K, V> = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
        let node = unsafe { inode.node.load(SeqCst).as_ref().unwrap() };
        node.count()
    }

    /// Return whether map is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K, V, H> Map<K, V, H>
where
    K: Clone,
    V: Clone,
    H: Clone,
{
    /// Call this method after all other concurrnet instances have been
    /// dropped.
    ///
    /// * There shall be no tomb-nodes.
    /// * There shall be no empty trie-nodes that is not root.
    /// * There shall be no trie-nodes with childs.len() > 16.
    /// * There shall be no list-node with items.len() < 2
    /// * All list-nodes must be at 9th level.
    pub fn validate(&self) -> Stats {
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

        debug_assert!(
            stats.n_tombs == 0,
            "unexpected tomb nodes {}",
            stats.n_tombs
        );

        stats
    }

    #[cfg(test)]
    pub fn collisions(&self)
    where
        K: Hash + Debug,
        V: Debug,
        H: BuildHasher,
    {
        unsafe {
            self.root
                .load(SeqCst)
                .as_ref()
                .unwrap()
                .collisions(&self.hash_builder)
        };
    }
}

impl<K, V, H> Map<K, V, H>
where
    K: Clone,
    V: Clone,
{
    #[cfg(any(test, feature = "perf"))]
    pub fn print(&self)
    where
        K: Debug,
        V: Debug,
    {
        let access_log = self.access_log.iter().map(|e| e.load(SeqCst));
        println!("Map<{},{:?}>", self.epoch.load(SeqCst), access_log);

        unsafe { self.root.load(SeqCst).as_ref().unwrap().print("  ") };
    }

    #[cfg(any(test, feature = "perf"))]
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

impl<K, V> Child<K, V>
where
    K: Clone,
    V: Clone,
{
    fn new_leaf(leaf: Item<K, V>, cas: &mut gc::Cas<K, V>) -> *mut Child<K, V> {
        let mut child = cas.alloc_child();

        *child = Child::Leaf(leaf);

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

    fn is_tomb_node(&self) -> bool {
        match self {
            Child::Leaf(_) => false,
            Child::Deep(inode) => {
                let node = unsafe { inode.node.load(SeqCst).as_ref().unwrap() };
                matches!(node, Node::Tomb { .. })
            }
            Child::None => unreachable!(),
        }
    }
}

impl<K, V> Child<K, V> {
    fn dropped(child: *mut Child<K, V>) {
        let child = unsafe { Box::from_raw(child) };
        match child.as_ref() {
            Child::Leaf(_item) => (),
            Child::Deep(inode) => Node::dropped(inode.node.load(SeqCst)),
            Child::None => unreachable!(),
        }
    }
}

impl<K, V> Node<K, V> {
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
                *bmp |= 1 << w;
            }
            Node::Tomb { .. } | Node::List { .. } => (),
        }
    }

    #[inline]
    fn hamming_reset(&mut self, w: u8) {
        match self {
            Node::Trie { bmp, .. } => {
                *bmp &= !(1 << w);
            }
            Node::Tomb { .. } | Node::List { .. } => (),
        }
    }

    fn as_value<'a, Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: PartialEq + Hash + ?Sized,
    {
        match self {
            Node::List { items } => get_from_list(key, items),
            Node::Tomb { item } => match item {
                Some(item) => {
                    if item.key.borrow() == key {
                        Some(&item.value)
                    } else {
                        None
                    }
                }
                None => None,
            },
            Node::Trie { .. } => unreachable!(),
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
                        Child::None => unreachable!(),
                    }
                }
                len
            }
            Node::List { items } => items.len(),
            Node::Tomb { item: None } => 0,
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
    K: Clone,
    V: Clone,
{
    #[cfg(any(test, feature = "perf"))]
    fn print(&self, prefix: &str)
    where
        K: Debug,
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
                        Child::None => unreachable!(),
                    }
                }
            }
            Node::Tomb { item: None } => println!("{}Node::Tomb-None", prefix),
            Node::Tomb { item } => {
                println!("{}Node::Tomb", prefix);
                let prefix = prefix.to_string() + "  ";
                item.as_ref().unwrap().print(&prefix);
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
            Node::Tomb { .. } => unreachable!(),
            Node::Trie { childs, .. } => {
                let nc = childs.len();
                if depth > 1 {
                    debug_assert!(nc > 0, "unexpected node.trie n:{}", nc);
                } else {
                    debug_assert!(nc <= 16, "unexpected node.trie n:{}", nc);
                }
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
                        Child::None => unreachable!(),
                    }
                }
            }
            Node::List { items } => {
                debug_assert!(items.len() > 1, "unexpected node.list n:{}", items.len());
                debug_assert!(depth == 9, "unexpected node.list depth:{}", depth);
                stats.n_lists += 1;
                stats.n_items += items.len();
                stats.n_mem += items.capacity() * mem::size_of::<Item<K, V>>();
            }
        }

        stats
    }

    #[cfg(test)]
    fn collisions<H>(&self, hb: &H)
    where
        K: Hash + Debug,
        H: BuildHasher,
    {
        match self {
            Node::Trie { childs, .. } => {
                for child in childs {
                    match unsafe { child.load(SeqCst).as_ref().unwrap() } {
                        Child::Leaf(_) => (),
                        Child::Deep(inode) => inode.collisions(hb),
                        Child::None => unreachable!(),
                    }
                }
            }
            Node::Tomb { .. } => (),
            Node::List { items } => {
                for item in items {
                    let hasher = hb.build_hasher();
                    println!("key:{:?},{}", item.key, key_to_hash32(&item.key, hasher))
                }
            }
        }
    }
}

impl<K, V> Node<K, V>
where
    K: Clone,
    V: Clone,
{
    fn new_bi_list(
        item: Item<K, V>,
        leaf: &Item<K, V>,
        cas: &mut gc::Cas<K, V>,
    ) -> *mut Node<K, V> {
        let mut node = cas.alloc_node('l');

        match node.as_mut() {
            Node::List { items } => {
                items.clear();
                // unsafe { items.set_len(2) }; // **IMPORTANT**
                items.push(item);
                items.push(leaf.clone());
                #[cfg(freature = "compact")]
                items.shrink_to_fit();
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
        cas: &mut gc::Cas<K, V>,
    ) -> (*mut Node<K, V>, Option<V>)
    where
        K: PartialEq,
    {
        let mut node = cas.alloc_node('l');

        let old_value = match node.as_mut() {
            Node::List { items } => {
                items.clear();
                items.extend_from_slice(olds);
                let old_value = update_into_list(item, items);
                #[cfg(feature = "compact")]
                items.shrink_to_fit();
                old_value
            }
            _ => unreachable!(),
        };

        let node_ptr = Box::leak(node);
        cas.free_on_fail(gc::Mem::Node(node_ptr));
        (node_ptr, old_value)
    }

    fn new_list_without(
        olds: &[Item<K, V>],
        i: usize,
        cas: &mut gc::Cas<K, V>,
    ) -> *mut Node<K, V> {
        let mut node = cas.alloc_node('l');

        match node.as_mut() {
            Node::List { items } => {
                items.clear();
                items.extend_from_slice(&olds[..i]);
                items.extend_from_slice(&olds[i + 1..]); // skip i
                #[cfg(feature = "compact")]
                items.shrink_to_fit();
            }
            _ => unreachable!(),
        }

        let node_ptr = Box::leak(node);
        cas.free_on_fail(gc::Mem::Node(node_ptr));
        node_ptr
    }

    fn new_tomb(nitem: &Item<K, V>, cas: &mut gc::Cas<K, V>) -> *mut Node<K, V> {
        let mut node = cas.alloc_node('b');

        *node = Node::Tomb {
            item: Some(nitem.clone()),
        };

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
        match pairs.first() {
            None => Node::new_bi_list(item, leaf, op.cas),
            Some((w1, w2)) if *w1 == *w2 => {
                // println!("new subtrie:{:x},{:x}", w1, w2);
                let mut node = op.cas.alloc_node('t');

                match node.as_mut() {
                    Node::Trie { childs, .. } => {
                        let node = Self::new_subtrie(item, leaf, &pairs[1..], op);
                        childs.clear();
                        childs.insert(0, AtomicPtr::new(Child::new_deep(node, op.cas)));
                        #[cfg(feature = "compact")]
                        childs.shrink_to_fit();
                    }
                    _ => unreachable!(),
                };
                node.hamming_set(*w1);

                let node_ptr = Box::leak(node);
                op.cas.free_on_fail(gc::Mem::Node(node_ptr));
                node_ptr
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
                        let n = match hamming_distance(*w2, *bmp) {
                            Distance::Insert(n) => n,
                            Distance::Set(_) => unreachable!(),
                        };
                        let leaf = leaf.clone();
                        childs.insert(n, AtomicPtr::new(Child::new_leaf(leaf, op.cas)));
                        #[cfg(feature = "compact")]
                        childs.shrink_to_fit();
                    }
                    _ => unreachable!(),
                };
                node.hamming_set(*w2);

                let node_ptr = Box::leak(node);
                op.cas.free_on_fail(gc::Mem::Node(node_ptr));
                node_ptr
            }
        }
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
            Node::Tomb { .. } => unreachable!(),
            Node::Trie { .. } => unreachable!(),
        }
    }

    fn ins_child(leaf: Item<K, V>, w: u8, n: usize, op: CasOp<K, V>) -> CasRc<()> {
        let new_child_ptr = Child::new_leaf(leaf, op.cas);

        let mut node = op.cas.alloc_node('t');
        node.trie_copy_from(unsafe { op.old.as_ref().unwrap() });

        match node.as_mut() {
            Node::Trie { childs, .. } => {
                childs.insert(n, AtomicPtr::new(new_child_ptr));
                #[cfg(feature = "compact")]
                childs.shrink_to_fit();
            }
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

        // println!("set_child");
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
                    Child::None => unreachable!(),
                };
                childs[n] = AtomicPtr::new(new_child_ptr);
            }
            _ => unreachable!(),
        }

        let new = Box::leak(node);

        // println!("leaf_to_list");
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

        let mut new_node = op.cas.alloc_node('t');
        new_node.trie_copy_from(unsafe { op.old.as_ref().unwrap() });

        match new_node.as_mut() {
            Node::Trie { childs, .. } => {
                childs[n] = AtomicPtr::new(Child::new_deep(node, op.cas));
            }
            _ => unreachable!(),
        }

        let new = Box::leak(new_node);

        // println!("set_trie_child");
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
            Node::List { items } if items.len() == 1 => unreachable!(),
            Node::List { items } if items.len() == 2 => {
                let j = [1, 0][n];
                let new = Node::new_tomb(&items[j], op.cas);
                (true, new, items[n].value.clone())
            }
            Node::List { items } if !items.is_empty() => (
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

    fn remove_child1(n: usize, op: CasOp<K, V>) -> CasRc<()> {
        // println!("remove_child1 w:{:x} n:{}", w, n);

        let old_child_ptr = match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { childs, .. } if childs.len() == 1 => childs[n].load(SeqCst),
            _ => unreachable!(),
        };

        let mut node = op.cas.alloc_node('b');
        match node.as_mut() {
            Node::Tomb { item } => *item = None,
            _ => unreachable!(),
        }

        let new = Box::leak(node);

        // println!("remove_child1");
        op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
        op.cas.free_on_fail(gc::Mem::Node(new));
        op.cas.free_on_pass(gc::Mem::Node(op.old));
        if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
            CasRc::Ok(())
        } else {
            CasRc::Retry
        }
    }

    fn remove_child2(m: &Item<K, V>, op: CasOp<K, V>) -> CasRc<()> {
        let mut node = op.cas.alloc_node('b');
        *node = Node::Tomb {
            item: Some(m.clone()),
        };

        let new = Box::leak(node);

        op.cas.free_on_fail(gc::Mem::Node(new));
        op.cas.free_on_pass(gc::Mem::Node(op.old));
        if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
            CasRc::Ok(())
        } else {
            CasRc::Retry
        }
    }

    fn remove_child(w: u8, n: usize, op: CasOp<K, V>) -> CasRc<()> {
        // println!("remove_child w:{:x} n:{}", w, n);

        let old_child_ptr = match unsafe { op.old.as_ref().unwrap() } {
            Node::Trie { childs, .. } => childs[n].load(SeqCst),
            _ => unreachable!(),
        };

        let mut node = op.cas.alloc_node('t');
        node.trie_copy_from(unsafe { op.old.as_ref().unwrap() });

        match node.as_mut() {
            Node::Trie { childs, .. } => {
                childs.remove(n);
                #[cfg(feature = "compact")]
                childs.shrink_to_fit();
            }
            _ => unreachable!(),
        };
        node.hamming_reset(w);

        let new = Box::leak(node);

        // println!("remove_child");
        op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
        op.cas.free_on_fail(gc::Mem::Node(new));
        op.cas.free_on_pass(gc::Mem::Node(op.old));
        if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
            CasRc::Ok(())
        } else {
            CasRc::Retry
        }
    }

    fn compact_trie_from(
        w: u8,
        n: usize,
        depth: usize,
        op: CasOp<K, V>,
    ) -> (bool, CasRc<()>) {
        let mut node = op.cas.alloc_node('t');
        node.trie_copy_from(unsafe { op.old.as_ref().unwrap() });

        match node.as_mut() {
            Node::Trie { childs, .. } => {
                let old_child_ptr = childs[n].load(SeqCst);
                match unsafe { old_child_ptr.as_ref().unwrap() } {
                    Child::Deep(next_inode) => {
                        let next_node_ptr = next_inode.node.load(SeqCst);
                        match unsafe { next_node_ptr.as_ref().unwrap() } {
                            Node::Tomb { item: None } => {
                                // println!("compact_trie_from");
                                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                                op.cas.free_on_pass(gc::Mem::Node(next_node_ptr));
                                childs.remove(n);
                                node.hamming_reset(w)
                            }
                            Node::Tomb { item: Some(item) } => {
                                // println!("compact_trie_from");
                                op.cas.free_on_pass(gc::Mem::Child(old_child_ptr));
                                op.cas.free_on_pass(gc::Mem::Node(next_node_ptr));
                                let new_ptr = Child::new_leaf(item.clone(), op.cas);
                                childs[n] = AtomicPtr::new(new_ptr);
                            }
                            _ => unreachable!(),
                        }
                    }
                    Child::Leaf(_) | Child::None => unreachable!(),
                }
            }
            _ => unreachable!(),
        };

        // self compact only for non-root nodes.
        let compact = match node.as_mut() {
            Node::Trie { childs, .. } => match childs.len() {
                0 if depth > 1 => {
                    *node = Node::Tomb { item: None };
                    true
                }
                1 if depth > 1 => {
                    let other_child_ptr = childs[0].load(SeqCst);
                    match unsafe { other_child_ptr.as_ref().unwrap() } {
                        Child::Leaf(m) => {
                            // println!("compact_trie_from");
                            op.cas.free_on_pass(gc::Mem::Child(other_child_ptr));
                            let item = Some(m.clone());
                            *node = Node::Tomb { item };
                            true
                        }
                        _ => {
                            #[cfg(feature = "compact")]
                            childs.shrink_to_fit();
                            false
                        }
                    }
                }
                _ => {
                    #[cfg(feature = "compact")]
                    childs.shrink_to_fit();
                    false
                }
            },
            _ => unreachable!(),
        };

        let new = Box::leak(node);

        op.cas.free_on_fail(gc::Mem::Node(new));
        op.cas.free_on_pass(gc::Mem::Node(op.old));
        if op.cas.swing(op.epoch, &op.inode.node, op.old, new) {
            (compact, CasRc::Ok(()))
        } else {
            (compact, CasRc::Retry)
        }
    }
}

impl<K, V, H> Map<K, V, H> {
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + Hash + ?Sized,
        H: BuildHasher,
    {
        let seqno = self.epoch.load(SeqCst);
        self.access_log[self.id].store(seqno | ENTER_MASK, SeqCst);

        let ws = {
            let hasher = self.hash_builder.build_hasher();
            slots(key_to_hash32(key, hasher))
        };
        let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
        let mut wss = &ws[..];
        // println!("{}", format_ws!("get outer ws:{:?}", wss));

        let res = loop {
            let old = inode.node.load(SeqCst);
            let node = unsafe { old.as_ref().unwrap() };

            let w = match wss.first() {
                Some(w) => *w,
                None => break node.as_value(key).cloned(),
            };
            wss = &wss[1..];
            // println!("get loop w:{:x}", w);

            inode = match node {
                Node::Trie { bmp, childs } => {
                    let hd = hamming_distance(w, *bmp);
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
                                Child::None => unreachable!(),
                            }
                        }
                    }
                }
                Node::List { .. } => unreachable!(),
                Node::Tomb { item } => match item {
                    Some(m) if m.key.borrow() == key => break Some(m.value.clone()),
                    _ => break None,
                },
            }
        };

        self.access_log[self.id].store(seqno, SeqCst);
        res
    }

    pub fn get_with<Q, F, T>(&self, key: &Q, mut callb: F) -> Option<T>
    where
        K: Borrow<Q>,
        Q: PartialEq + Hash + ?Sized,
        H: BuildHasher,
        F: FnMut(&V) -> T,
    {
        let seqno = self.epoch.load(SeqCst);
        self.access_log[self.id].store(seqno | ENTER_MASK, SeqCst);

        let ws = {
            let hasher = self.hash_builder.build_hasher();
            slots(key_to_hash32(key, hasher))
        };
        let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
        let mut wss = &ws[..];
        // println!("{}", format_ws!("get outer ws:{:?}", wss));
        // println!("get_with seqno:{}", seqno);

        let res = loop {
            let old = inode.node.load(SeqCst);
            let node = unsafe { old.as_ref().unwrap() };

            let w = match wss.first() {
                Some(w) => *w,
                None => break node.as_value(key).map(callb),
            };
            wss = &wss[1..];
            // println!("get loop w:{:x}", w);

            inode = match node {
                Node::Trie { bmp, childs } => {
                    let hd = hamming_distance(w, *bmp);
                    // println!("get loop bmp:{:x} {:?}", bmp, hd);
                    match hd {
                        Distance::Insert(_) => break None,
                        Distance::Set(n) => {
                            let ptr = childs[n].load(SeqCst);
                            match unsafe { ptr.as_ref().unwrap() } {
                                Child::Deep(next_inode) => next_inode,
                                Child::Leaf(item) if item.key.borrow() == key => {
                                    break Some(callb(&item.value));
                                }
                                Child::Leaf(_) => break None,
                                Child::None => unreachable!(),
                            }
                        }
                    }
                }
                Node::List { .. } => unreachable!(),
                Node::Tomb { item } => match item {
                    Some(m) if m.key.borrow() == key => break Some(callb(&m.value)),
                    _ => break None,
                },
            }
        };

        self.access_log[self.id].store(seqno, SeqCst);
        res
    }

    pub fn set(&mut self, key: K, value: V) -> Option<V>
    where
        K: Clone + PartialEq + Hash,
        V: Clone,
        H: BuildHasher,
    {
        let (seqno, res) = self.do_set(key, value);
        if self.gc_count == 0 {
            let seqno = gc_epoch!(self.access_log, seqno);
            if seqno < u64::MAX {
                self.cas.garbage_collect(seqno)
            }
            self.gc_count = self.gc_period; // reload
        }
        self.gc_count = self.gc_count.saturating_sub(1);

        res
    }

    fn do_set(&mut self, key: K, value: V) -> (u64, Option<V>)
    where
        K: Clone + PartialEq + Hash,
        V: Clone,
        H: BuildHasher,
    {
        let seqno = self.epoch.load(SeqCst);
        self.access_log[self.id].store(seqno | ENTER_MASK, SeqCst);

        let ws = {
            let hasher = self.hash_builder.build_hasher();
            slots(key_to_hash32(&key, hasher))
        };
        let res = 'retry: loop {
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
            let mut wss = &ws[..];
            // println!("set try key:{:?} {}", key, format_ws!("{:?}", ws));

            loop {
                let old: *mut Node<K, V> = inode.node.load(SeqCst);
                let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => match node {
                        Node::Tomb { .. } => continue 'retry,
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
                    Node::Trie { bmp, .. } => match hamming_distance(w, *bmp) {
                        Distance::Insert(n) => {
                            // println!("set loop insert bmp:{:x} {}", bmp, n);
                            let op = generate_op!(self, inode, old);
                            let item = (key.clone(), value.clone()).into();
                            match Node::ins_child(item, w, n, op) {
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

                inode = match unsafe { node.get_child(n).as_ref().unwrap() } {
                    Child::Deep(next_node) => {
                        // println!("set loop next level {:?}", key);
                        next_node
                    }
                    Child::Leaf(ot) if ot.key == key => {
                        let op = generate_op!(self, inode, old);
                        // println!("set loop 1");

                        let item = (key.clone(), value.clone()).into();
                        match Node::set_child(item, n, op) {
                            CasRc::Ok(_) => break 'retry Some(ot.value.clone()),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(_) if wss.is_empty() => {
                        let op = generate_op!(self, inode, old);
                        // println!("set loop 2");

                        match Node::leaf_to_list(key.clone(), &value, n, op) {
                            CasRc::Ok(_) => break 'retry None,
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(leaf) => {
                        let hash_lk = {
                            let hasher = self.hash_builder.build_hasher();
                            key_to_hash32(&leaf.key, hasher)
                        };

                        let mut op = generate_op!(self, inode, old);
                        let mut scratch = [(0_u8, 0_u8); 8];
                        let xs = subtrie_zip(hash_lk, wss, &mut scratch);
                        // println!("set loop 3");

                        let item: Item<K, V> = (key.clone(), value.clone()).into();
                        let node_ptr = Node::new_subtrie(item, leaf, xs, &mut op);

                        match Node::set_trie_child(node_ptr, n, op) {
                            CasRc::Ok(_) => break 'retry None,
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::None => unreachable!(),
                }
            }
        };

        self.access_log[self.id].store(seqno, SeqCst);
        self.epoch.fetch_add(1, SeqCst);

        (seqno, res)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Clone + Borrow<Q>,
        V: Clone,
        Q: PartialEq + Hash + ?Sized,
        H: BuildHasher,
    {
        let (seqno, compact, res) = self.do_remove(key);
        if compact {
            self.do_compact(key)
        }
        if self.gc_count == 0 {
            let seqno = gc_epoch!(self.access_log, seqno);
            if seqno < u64::MAX {
                self.cas.garbage_collect(seqno)
            }
            self.gc_count = self.gc_period; // reload
        }
        self.gc_count = self.gc_count.saturating_sub(1);

        res
    }

    fn do_remove<Q>(&mut self, key: &Q) -> (u64, bool, Option<V>)
    where
        K: Clone + Borrow<Q>,
        V: Clone,
        Q: PartialEq + Hash + ?Sized,
        H: BuildHasher,
    {
        let seqno = self.epoch.load(SeqCst);
        self.access_log[self.id].store(seqno | ENTER_MASK, SeqCst);

        let ws = {
            let hasher = self.hash_builder.build_hasher();
            slots(key_to_hash32(key, hasher))
        };
        let (compact, res) = 'retry: loop {
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
            let mut wss = &ws[..];
            // println!("remove try key:{:?} {}", key, format_ws!("{:?}", ws));

            let mut depth = 0;
            loop {
                depth += 1;
                let old: *mut Node<K, V> = inode.node.load(SeqCst);
                let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => match node {
                        Node::List { items } if items.len() < 2 => unreachable!(),
                        Node::List { items } => match has_key(items, key) {
                            Some(n) => {
                                let op = generate_op!(self, inode, old);
                                match Node::remove_from_list(n, op) {
                                    (_, CasRc::Retry) => continue 'retry,
                                    (comp, CasRc::Ok(ov)) => break 'retry (comp, ov),
                                }
                            }
                            None => break 'retry (false, None),
                        },
                        Node::Tomb { item } => match item {
                            Some(m) if m.key.borrow() == key => continue 'retry,
                            _ => break 'retry (false, None),
                        },
                        Node::Trie { .. } => unreachable!(),
                    },
                };

                wss = &wss[1..];
                // println!("do_remove w:{:x}", w);

                let (n, _bmp, childs) = match node {
                    Node::Trie { bmp, childs } => match childs.len() {
                        0 => break 'retry (false, None),
                        _ => match hamming_distance(w, *bmp) {
                            Distance::Insert(_) => break 'retry (false, None),
                            Distance::Set(n) => (n, bmp, childs),
                        },
                    },
                    Node::Tomb { item } => match item {
                        Some(item) if item.key.borrow() == key => continue 'retry,
                        _ => break 'retry (false, None),
                    },
                    Node::List { .. } => unreachable!(),
                };
                // println!("do_remove n:{} bmp:{:x}", n, _bmp);

                let ocp = childs[n].load(SeqCst);
                inode = match unsafe { ocp.as_ref().unwrap() } {
                    Child::Deep(next_inode) => next_inode,
                    Child::Leaf(item) if item.key.borrow() == key && depth == 1 => {
                        // println!("remove1 old value {:?}", ov);

                        // avoid tombification of node, root node should always be
                        // a trie and can contain empty childs.
                        let op = generate_op!(self, inode, old);
                        let ov = item.value.clone();
                        match Node::remove_child(w, n, op) {
                            CasRc::Ok(_) => break 'retry (false, Some(ov)),
                            CasRc::Retry => continue 'retry,
                        }
                    }
                    Child::Leaf(item) if item.key.borrow() == key => {
                        let ov = item.value.clone();
                        let (compact, res) = match childs.len() {
                            0 => unreachable!(),
                            1 => {
                                // println!("remove2 old value {:?}", ov);

                                let op = generate_op!(self, inode, old);
                                debug_assert!(n == 0, "unexpected {}", n);
                                (true, Node::remove_child1(n, op))
                            }
                            2 => {
                                let op = generate_op!(self, inode, old);

                                let j = [1, 0][n];
                                let lcp = childs[j].load(SeqCst);
                                match unsafe { lcp.as_ref().unwrap() } {
                                    Child::Leaf(m) => {
                                        // println!("remove3 old value");

                                        op.cas.free_on_pass(gc::Mem::Child(lcp));
                                        op.cas.free_on_pass(gc::Mem::Child(ocp));
                                        (true, Node::remove_child2(m, op))
                                    }
                                    Child::Deep(_) => {
                                        // println!("remove4 old value {:?}", ov);

                                        (false, Node::remove_child(w, n, op))
                                    }
                                    Child::None => unreachable!(),
                                }
                            }
                            _ => {
                                let op = generate_op!(self, inode, old);
                                (false, Node::remove_child(w, n, op))
                            }
                        };

                        // println!("remove old value {:?}", ov);
                        match (compact, res) {
                            (compact, CasRc::Ok(_)) => break 'retry (compact, Some(ov)),
                            (_, CasRc::Retry) => continue 'retry,
                        }
                    }
                    Child::Leaf(_) => break 'retry (false, None),
                    Child::None => unreachable!(),
                }
            }
        };

        self.access_log[self.id].store(seqno, SeqCst);
        self.epoch.fetch_add(1, SeqCst);

        (seqno, compact, res)
    }

    fn do_compact<Q>(&mut self, key: &Q)
    where
        K: Clone + Borrow<Q>,
        V: Clone,
        Q: Hash + ?Sized,
        H: BuildHasher,
    {
        let seqno = self.epoch.load(SeqCst);
        self.access_log[self.id].store(seqno | ENTER_MASK, SeqCst);

        let ws = {
            let hasher = self.hash_builder.build_hasher();
            slots(key_to_hash32(key, hasher))
        };
        'retry: loop {
            let mut inode = unsafe { self.root.load(SeqCst).as_ref().unwrap() };
            let mut wss = &ws[..];
            //println!(
            //    "do_compact key:{:?} {} retry:{}",
            //    key,
            //    format_ws!("{:?}", wss),
            //    retry
            //);

            let mut depth = 0;
            loop {
                depth += 1;
                let old: *mut Node<K, V> = inode.node.load(SeqCst);
                let node: &Node<K, V> = unsafe { old.as_ref().unwrap() };

                let w = match wss.first() {
                    Some(w) => *w,
                    None => match node {
                        Node::Tomb { .. } => continue 'retry,
                        Node::List { items } if items.len() < 2 => unreachable!(),
                        Node::List { .. } => break 'retry,
                        Node::Trie { .. } => unreachable!(),
                    },
                };
                wss = &wss[1..];
                // println!("do_compact w:{:x}", w);

                let (n, child) = match node {
                    Node::Tomb { .. } => continue 'retry,
                    Node::Trie { bmp, childs } => {
                        let hd = hamming_distance(w, *bmp);
                        let n = match hd {
                            Distance::Insert(_) => break 'retry,
                            Distance::Set(n) => n,
                        };
                        (n, unsafe { childs[n].load(SeqCst).as_ref().unwrap() })
                    }
                    Node::List { .. } => unreachable!(),
                };

                inode = match node {
                    Node::Trie { .. } if child.is_tomb_node() => {
                        let op = generate_op!(self, inode, old);
                        match Node::compact_trie_from(w, n, depth, op) {
                            (true, CasRc::Ok(_)) => continue 'retry,
                            (false, CasRc::Ok(_)) => break 'retry,
                            (_, CasRc::Retry) => continue 'retry,
                        }
                    }
                    Node::Trie { .. } => match child {
                        Child::Leaf(_) => break 'retry,
                        Child::Deep(inode) => inode,
                        Child::None => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }
        }

        self.access_log[self.id].store(seqno, SeqCst);
        self.epoch.fetch_add(1, SeqCst);
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
    cas: &'a mut gc::Cas<K, V>,
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
    let dist = (x ^ y).count_ones() as usize;

    match bmp & posn {
        0 => Distance::Insert(dist),
        _ => Distance::Set(dist),
    }
}

fn key_to_hash32<K, H>(key: &K, mut hasher: H) -> u32
where
    K: Hash + ?Sized,
    H: Hasher,
{
    key.hash(&mut hasher);
    let code: u64 = hasher.finish();
    (((code >> 32) ^ code) & 0xFFFFFFFF) as u32
}

fn slots(key: u32) -> [u8; 8] {
    let mut arr = [0_u8; 8];
    for (i, item) in arr.iter_mut().enumerate() {
        *item = ((key >> (i * 4)) & SLOT_MASK) as u8;
    }
    arr
}

fn subtrie_zip<'a>(hash: u32, wss: &[u8], out: &'a mut [(u8, u8); 8]) -> &'a [(u8, u8)] {
    let ls = slots(hash);
    let n = wss.len();
    let off = 8 - n;
    for i in 0..n {
        out[i] = (wss[i], ls[off + i])
    }
    &out[..n]
}

fn get_from_list<'a, K, V, Q>(key: &Q, items: &'a [Item<K, V>]) -> Option<&'a V>
where
    K: Borrow<Q>,
    Q: PartialEq + ?Sized,
{
    items
        .iter()
        .find(|x| x.key.borrow() == key)
        .map(|x| &x.value)
}

fn update_into_list<K, V>(item: Item<K, V>, items: &mut Vec<Item<K, V>>) -> Option<V>
where
    K: PartialEq,
    V: Clone,
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

fn has_key<K, V, Q>(items: &[Item<K, V>], key: &Q) -> Option<usize>
where
    K: Borrow<Q>,
    Q: PartialEq + ?Sized,
{
    items
        .iter()
        .enumerate()
        .find(|(_, x)| x.key.borrow() == key)
        .map(|res| res.0)
}

/// Map statistics.
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
#[path = "dash1_test.rs"]
mod dash1_test;

#[cfg(test)]
#[path = "dash2_test.rs"]
mod dash2_test;

#[cfg(test)]
#[path = "map_test.rs"]
mod map_test;
