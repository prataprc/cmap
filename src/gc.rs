//! Module implement epochal garbage collection algorithm.
//!
//! * Epoch is maintained as type [u64], a sequence number that monotonically increases
//!   for every access into the trie structure, across all threads.
//! * Epochal garbage collection is implemented for `Node` and `Child` types, variants of
//!   `Mem` enum.
//! * An alter-ego of `Mem` type is maintained as `OwnedMem`, which holds the ownership
//!   to allocated memory, instead of just the pointer.
//! * `Reclaim` type maintains a list of allocations which was allocated at a known epoch,
//!   and these allocations can be freed once all the accessing threads has moved past
//!   the epoch.
//!
//! **Swing**
//!
//! Swing is `compare_exchange` atomic operation done to trie nodes. Success and Failure
//! of the operation is tightly coupled with epochal-garbage-collection.
//!
//! ```ignore
//!     alloc -> swing --(success)--> reclaim-old-allocs
//!                |
//!                +-----(failure)--> free-new-allocs
//! ```
//!
//! **Garbage Collection**
//!
//! GC is invoked periodically along the trie-execution-path. The general idea is to
//! check whether any of the reclaim entry in the `Cas::reclaim` list less than the
//! current `gc_epoch`.
//!
//! A note on current `epoch`. It is a 64-bit sequence number atomically shared between
//! every clone of the Map instance. This value is initialised with `1` and monotonically
//! increases when a `set` or `remove` access into the trie-structure _complete_.
//!
//! A note on `access_log`. It is an array of 64-bit sequence number that is atomically
//! updated with current `epoch` for every `get`, `set`, `remove`. Most significant
//! bit of the access log specifies whether the access is on-going and the remaining bits
//! specifying the `epoch` at which the access started.
//!
//! A note on `gc_epoch`. It is a 64-bit sequence number that is computed, for each
//! gc-period, along the [Map::set] and [Map::remove] execution path. Additionally
//! it is computed when the map is dropped. A per-thread `access_log` is maintained and
//! shared, lockless, across every clone of the [Map].
//!
//!
//! **Allocation pools**
//!
//! * **child_pool**, to recycle enum `Child` type. They hold Leaf-Item or None,
//!   or an atomic-pointer to Node.
//! * **node_trie_pool**, **node_list_pool**, **node_tomb_pool**, to recycle enum `Node`
//!   type. Separate pools are maintained for each vartiant, to optimize on vector
//!   allocation.
//! * **reclaim_pool**, to recycle `Reclaim` type.
//!
//! **Validation checks**
//!
//! * When Cas instance is dropped, assert length of self.older, self.newer,
//!   self.reclaims list as ZERO.
//!

use std::{
    fmt, ptr, result,
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        Arc,
    },
};

#[allow(unused_imports)]
use crate::map::Map;
use crate::{map::Child, map::Node};

pub(crate) const MAX_POOL_SIZE: usize = 1024;

// CAS operation

pub(crate) struct Cas<K, V> {
    #[allow(clippy::vec_box)]
    reclaims: Vec<Box<Reclaim<K, V>>>,
    older: Vec<OwnedMem<K, V>>,
    newer: Vec<OwnedMem<K, V>>,

    child_pool: Vec<Box<Child<K, V>>>,
    node_trie_pool: Vec<Box<Node<K, V>>>,
    node_list_pool: Vec<Box<Node<K, V>>>,
    node_tomb_pool: Vec<Box<Node<K, V>>>,
    #[allow(clippy::vec_box)]
    reclaim_pool: Vec<Box<Reclaim<K, V>>>,

    n_allocs: usize,
    n_frees: usize,
}

impl<K, V> Drop for Cas<K, V> {
    fn drop(&mut self) {
        debug_assert!(
            self.older.is_empty(),
            "invariant Cas::older should be ZERO on drop"
        );
        debug_assert!(
            self.newer.is_empty(),
            "invariant Cas::newer should be ZERO on drop"
        );
        debug_assert!(
            self.reclaims.is_empty(),
            "invariant Cas::reclaims should be ZERO on drop"
        );

        //#[cfg(test)]
        //println!(
        //    "Dropping Cas pools:reclaims:{}, pools:({},{},{},{},{}) allocs:{}/{}",
        //    self.reclaims.len(),
        //    self.child_pool.len(),
        //    self.node_trie_pool.len(),
        //    self.node_list_pool.len(),
        //    self.node_tomb_pool.len(),
        //    self.reclaim_pool.len(),
        //    self.n_allocs,
        //    self.n_frees
        //);
    }
}

impl<K, V> Cas<K, V> {
    pub fn new() -> Self {
        Cas {
            reclaims: Vec::with_capacity(64),
            older: Vec::with_capacity(64),
            newer: Vec::with_capacity(64),

            child_pool: Vec::with_capacity(64),
            node_trie_pool: Vec::with_capacity(64),
            node_list_pool: Vec::with_capacity(64),
            node_tomb_pool: Vec::with_capacity(64),
            reclaim_pool: Vec::with_capacity(64),

            n_allocs: 0,
            n_frees: 0,
        }
    }

    pub fn to_pools_len(&self) -> usize {
        self.child_pool.len()
            + self.node_trie_pool.len()
            + self.node_list_pool.len()
            + self.node_tomb_pool.len()
            + self.reclaim_pool.len()
            + self.reclaims.len()
            + self.reclaims.iter().map(|r| r.items.len()).sum::<usize>()
    }

    pub fn to_alloc_count(&self) -> usize {
        self.n_allocs
    }

    pub fn to_free_count(&self) -> usize {
        self.n_frees
    }

    pub fn has_reclaims(&self) -> bool {
        !self.reclaims.is_empty()
    }

    // mark [Node] and [Child] allocations from the trie for garbage collection in case
    // the subsequent swing operation has passed.
    pub fn free_on_pass(&mut self, m: Mem<K, V>) {
        match m {
            Mem::Child(p) => unsafe {
                self.older.push(OwnedMem::Child(Box::from_raw(p)));
            },
            Mem::Node(p) => unsafe {
                self.older.push(OwnedMem::Node(Box::from_raw(p)));
            },
        }
    }

    // mark [Node] and [Child] allocations that needs to be inserted into the trie for
    // garbage collection in case the subsequent swing operaton has failed.
    pub fn free_on_fail(&mut self, m: Mem<K, V>) {
        match m {
            Mem::Child(p) => unsafe {
                self.newer.push(OwnedMem::Child(Box::from_raw(p)));
            },
            Mem::Node(p) => unsafe {
                self.newer.push(OwnedMem::Node(Box::from_raw(p)));
            },
        }
    }

    // allocate a node from one of the three node pools, based on the `variant`.
    pub fn alloc_node(&mut self, variant: char) -> Box<Node<K, V>> {
        match variant {
            'l' => match self.node_list_pool.pop() {
                Some(val) => val,
                None => {
                    self.n_allocs += 1;
                    Box::new(Node::List {
                        items: Vec::with_capacity(2), // **IMPORTANT**
                    })
                }
            },
            't' => match self.node_trie_pool.pop() {
                Some(val) => val,
                None => {
                    self.n_allocs += 1;
                    Box::new(Node::Trie {
                        bmp: 0,
                        childs: Vec::with_capacity(1),
                    })
                }
            },
            'b' => match self.node_tomb_pool.pop() {
                Some(val) => val,
                None => {
                    self.n_allocs += 1;
                    Box::new(Node::Tomb { item: None })
                }
            },
            _ => unreachable!(),
        }
    }

    // allocate a child from the child pool.
    pub fn alloc_child(&mut self) -> Box<Child<K, V>> {
        match self.child_pool.pop() {
            Some(val) => val,
            None => {
                self.n_allocs += 1;
                Box::new(Child::default())
            }
        }
    }

    // allocate reclaim instance from the reclaim pool.
    pub fn alloc_reclaim(&mut self) -> Box<Reclaim<K, V>> {
        match self.reclaim_pool.pop() {
            Some(val) => val,
            None => {
                self.n_allocs += 1;
                Box::new(Reclaim::default())
            }
        }
    }

    fn free_node(&mut self, mut node: Box<Node<K, V>>) {
        let pool = match node.as_mut() {
            Node::Trie { bmp, childs } => {
                *bmp = 0;
                childs.clear();
                &mut self.node_trie_pool
            }
            Node::List { items } => {
                items.clear();
                &mut self.node_list_pool
            }
            Node::Tomb { .. } => &mut self.node_tomb_pool,
        };
        if pool.len() < MAX_POOL_SIZE {
            pool.push(node)
        } else {
            self.n_frees += 1
        }
    }

    fn free_child(&mut self, child: Box<Child<K, V>>) {
        if self.child_pool.len() < MAX_POOL_SIZE {
            self.child_pool.push(child)
        } else {
            self.n_frees += 1
        }
    }

    fn free_reclaim(&mut self, reclaim: Box<Reclaim<K, V>>) {
        if self.reclaim_pool.len() < MAX_POOL_SIZE {
            self.reclaim_pool.push(reclaim)
        } else {
            self.n_frees += 1
        }
    }

    pub fn swing<T>(
        &mut self,
        epoch: &Arc<AtomicU64>,
        loc: &AtomicPtr<T>,
        old: *mut T,
        new: *mut T,
    ) -> bool {
        match loc.compare_exchange(old, new, SeqCst, SeqCst) {
            Ok(_) => {
                // drain the older allocations into the reclamation entry, and mark the
                // epoch.
                let r = {
                    let mut r = self.alloc_reclaim();
                    r.epoch = Some(epoch.load(SeqCst));
                    r.drain_items_from(&mut self.older);
                    r
                };
                self.reclaims.push(r);
                unsafe { self.newer.set_len(0) }; // leak newer values.
                true
            }
            Err(_) => {
                unsafe { self.older.set_len(0) }; // leak older values.
                while let Some(om) = self.newer.pop() {
                    match om {
                        OwnedMem::Child(val) => self.free_child(val),
                        OwnedMem::Node(val) => self.free_node(val),
                        OwnedMem::None => (),
                    }
                }
                false
            }
        }
    }

    pub fn garbage_collect(&mut self, gc_epoch: u64) {
        let n = self.reclaims.len();
        for i in (0..n).rev() {
            match self.reclaims[i].epoch {
                Some(epoch) if epoch < gc_epoch => {
                    let mut r = self.reclaims.remove(i);
                    while let Some(om) = r.items.pop() {
                        match om {
                            OwnedMem::Child(val) => self.free_child(val),
                            OwnedMem::Node(val) => self.free_node(val),
                            OwnedMem::None => (),
                        }
                    }
                    r.epoch = None;
                    self.free_reclaim(r);
                }
                Some(_) | None => (),
            }
        }
    }

    pub fn validate(&self) {
        let n = self.reclaims.len();
        debug_assert!(n < 512, "reclaims:{}", n);

        let n = self.older.len();
        debug_assert!(n < 512, "older:{}", n);

        let n = self.newer.len();
        debug_assert!(n < 512, "newer:{}", n);

        let n = self.child_pool.len();
        debug_assert!(n < 512, "child_pool:{}", n);

        let n = self.node_trie_pool.len();
        debug_assert!(n < 512, "node_trie_pool:{}", n);

        let n = self.node_list_pool.len();
        debug_assert!(n < 512, "node_list_pool:{}", n);

        let n = self.node_tomb_pool.len();
        debug_assert!(n < 512, "node_tomb_pool:{}", n);

        let n = self.reclaim_pool.len();
        debug_assert!(n < 512, "reclaim_pool:{}", n);
    }
}

pub(crate) struct Reclaim<K, V> {
    epoch: Option<u64>,
    items: Vec<OwnedMem<K, V>>,
}

impl<K, V> fmt::Debug for Reclaim<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        let items = self
            .items
            .iter()
            .map(|item| match item {
                OwnedMem::Child(_) => "child",
                OwnedMem::Node(_) => "node",
                OwnedMem::None => "None",
            })
            .collect::<Vec<&str>>()
            .join("");
        write!(f, "epoch:{:?} items:{:?}", self.epoch, items)
    }
}

impl<K, V> Default for Reclaim<K, V> {
    fn default() -> Self {
        Reclaim {
            epoch: None,
            items: Vec::with_capacity(2),
        }
    }
}

impl<K, V> Reclaim<K, V> {
    fn drain_items_from(&mut self, items: &mut Vec<OwnedMem<K, V>>) {
        debug_assert!(self.items.is_empty(), "reclaim items {}", self.items.len());

        self.items.reserve_exact(items.len());
        unsafe {
            self.items.set_len(items.len());
            ptr::copy(items.as_ptr(), self.items.as_mut_ptr(), items.len());
            items.set_len(0);
        };
    }
}

pub(crate) enum Mem<K, V> {
    Child(*mut Child<K, V>),
    Node(*mut Node<K, V>),
}

enum OwnedMem<K, V> {
    Child(Box<Child<K, V>>),
    Node(Box<Node<K, V>>),
    None,
}

impl<K, V> Default for OwnedMem<K, V> {
    fn default() -> Self {
        OwnedMem::None
    }
}
