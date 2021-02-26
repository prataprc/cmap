use std::sync::{
    atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
    Arc,
};

use crate::{map::Child, map::Item, map::Node};

// pub const EPOCH_PERIOD: time::Duration = time::Duration::from_millis(10);
pub const ENTER_MASK: u64 = 0x8000000000000000;
pub const EPOCH_MASK: u64 = 0x7FFFFFFFFFFFFFFF;
pub const MAX_POOL_SIZE: usize = 1024;

// CAS operation

pub struct Epoch {
    epoch: Arc<AtomicU64>,
    at: Arc<AtomicU64>,
}

impl Epoch {
    pub fn new(epoch: Arc<AtomicU64>, at: Arc<AtomicU64>) -> Epoch {
        at.store(epoch.load(SeqCst) | ENTER_MASK, SeqCst);
        Epoch { epoch, at }
    }
}

impl Drop for Epoch {
    fn drop(&mut self) {
        self.at.store(self.epoch.load(SeqCst), SeqCst);
    }
}

pub struct Cas<V> {
    reclaims: Vec<Box<Reclaim<V>>>,
    older: Vec<OwnedMem<V>>,
    newer: Vec<OwnedMem<V>>,

    child_pool: Vec<Box<Child<V>>>,
    node_tomb_pool: Vec<Box<Node<V>>>,
    node_list_pool: Vec<Box<Node<V>>>,
    node_trie_pool: Vec<Box<Node<V>>>,
    reclaim_pool: Vec<Box<Reclaim<V>>>,

    n_allocs: usize,
    n_frees: usize,
}

impl<V> Drop for Cas<V> {
    fn drop(&mut self) {
        assert!(
            self.older.len() == 0,
            "invariant Cas::older should be ZERO on drop"
        );
        assert!(
            self.newer.len() == 0,
            "invariant Cas::newer should be ZERO on drop"
        );
        assert!(
            self.reclaims.len() == 0,
            "invariant Cas::reclaims should be ZERO on drop"
        );
        #[cfg(test)]
        println!(
            "cas pools:{},({},{},{}),{} allocs:{}/{}",
            self.child_pool.len(),
            self.node_trie_pool.len(),
            self.node_list_pool.len(),
            self.node_item_pool.len(),
            self.reclaim_pool.len(),
            self.n_allocs,
            self.n_frees
        );
    }
}

impl<V> Cas<V> {
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

    pub fn has_reclaims(&self) -> bool {
        self.reclaims.len() > 0
    }

    pub fn free_on_pass(&mut self, m: Mem<V>) {
        match m {
            Mem::Child(ptr) => unsafe {
                self.older.push(OwnedMem::Child(Box::from_raw(ptr)));
            },
            Mem::Node(ptr) => unsafe {
                self.older.push(OwnedMem::Node(Box::from_raw(ptr)));
            },
        }
    }

    pub fn free_on_fail(&mut self, m: Mem<V>) {
        match m {
            Mem::Child(ptr) => unsafe {
                self.newer.push(OwnedMem::Child(Box::from_raw(ptr)));
            },
            Mem::Node(ptr) => unsafe {
                self.newer.push(OwnedMem::Node(Box::from_raw(ptr)));
            },
        }
    }

    pub fn alloc_node(&mut self, variant: char) -> Box<Node<V>> {
        match variant {
            'l' => match self.node_list_pool.pop() {
                Some(val) => val,
                None => {
                    self.n_allocs += 1;
                    Box::new(Node::List {
                        items: Vec::with_capacity(1),
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
                    Box::new(Node::Tomb {
                        item: Item::default(),
                    })
                }
            },
            _ => unreachable!(),
        }
    }

    pub fn alloc_child(&mut self) -> Box<Child<V>> {
        match self.child_pool.pop() {
            Some(val) => val,
            None => {
                self.n_allocs += 1;
                Box::new(Child::default())
            }
        }
    }

    pub fn alloc_reclaim(&mut self) -> Box<Reclaim<V>> {
        match self.reclaim_pool.pop() {
            Some(val) => val,
            None => {
                self.n_allocs += 1;
                Box::new(Reclaim::default())
            }
        }
    }

    pub fn free_node(&mut self, mut node: Box<Node<V>>) {
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

    pub fn free_child(&mut self, child: Box<Child<V>>) {
        if self.child_pool.len() < MAX_POOL_SIZE {
            self.child_pool.push(child)
        } else {
            self.n_frees += 1
        }
    }

    pub fn free_reclaim(&mut self, reclaim: Box<Reclaim<V>>) {
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
    ) -> bool
    where
        V: Clone,
    {
        if loc.compare_and_swap(old, new, SeqCst) == old {
            let r = {
                let mut r = self.alloc_reclaim();
                r.epoch = Some(epoch.load(SeqCst));
                r.items.clear();
                r.items.extend(self.older.drain(..)); // TODO: can we do memcpy ?
                r
            };
            self.reclaims.push(r);
            self.newer.drain(..).for_each(|m| m.leak());
            true
        } else {
            self.older.drain(..).for_each(|om| om.leak());
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
}

pub struct Reclaim<V> {
    epoch: Option<u64>,
    items: Vec<OwnedMem<V>>,
}

impl<V> Default for Reclaim<V> {
    fn default() -> Self {
        Reclaim {
            epoch: None,
            items: Vec::with_capacity(64),
        }
    }
}

pub enum Mem<V> {
    Child(*mut Child<V>),
    Node(*mut Node<V>),
}

enum OwnedMem<V> {
    Child(Box<Child<V>>),
    Node(Box<Node<V>>),
    None,
}

impl<V> Default for OwnedMem<V> {
    fn default() -> Self {
        OwnedMem::None
    }
}

impl<V> OwnedMem<V> {
    #[inline]
    fn leak(self) {
        match self {
            OwnedMem::Child(val) => {
                Box::leak(val);
            }
            OwnedMem::Node(val) => {
                Box::leak(val);
            }
            OwnedMem::None => (),
        }
    }
}

//pub fn gc_thread<V>(
//    epoch: Arc<AtomicU64>,
//    access_log: Arc<RwLock<Vec<Arc<AtomicU64>>>>,
//    rx: mpsc::Receiver<Reclaim<V>>,
//) {
//    let mut objs = vec![];
//
//    loop {
//        thread::sleep(EPOCH_PERIOD);
//        let (mut fresh, exit) = receive_blocks(&rx);
//        objs.append(&mut fresh);
//        if exit {
//            break;
//        }
//
//        let (gc, exited) = {
//            let log = access_log.read().expect("fail-lock");
//            let (epochs, exited): (Vec<u64>, bool) = {
//                let epochs: Vec<u64> = log.iter().map(|acc| acc.load(SeqCst)).collect();
//                (
//                    epochs
//                        .clone()
//                        .into_iter()
//                        .map(|el| {
//                            if el & ENTER_MASK == 0 {
//                                epoch.load(SeqCst)
//                            } else {
//                                el & EPOCH_MASK
//                            }
//                        })
//                        .collect(),
//                    epochs.into_iter().all(|epoch| epoch == 0),
//                )
//            };
//            let gc = match epochs.clone().into_iter().min() {
//                Some(gc) => gc,
//                None => continue,
//            };
//            (gc, exited)
//        };
//
//        if exited {
//            break;
//        }
//
//        let _n = objs.len();
//
//        let mut new_objs = vec![];
//        for obj in objs.into_iter() {
//            match obj.epoch {
//                Some(epoch) if epoch < gc => mem::drop(obj.items),
//                Some(_) | None => new_objs.push(obj),
//            }
//        }
//        objs = new_objs;
//
//        //#[cfg(test)] // TODO make it debug feature
//        //println!("garbage collected epoch:{} {}/{}", gc, _n, objs.len());
//    }
//
//    #[cfg(test)]
//    println!("exiting with pending allocs {}", objs.len());
//    mem::drop(objs);
//}

//fn receive_blocks<V>(rx: &mpsc::Receiver<Reclaim<V>>) -> (Vec<Reclaim<V>>, bool) {
//    let mut objs = vec![];
//    loop {
//        match rx.try_recv() {
//            Ok(recl) => objs.push(recl),
//            Err(mpsc::TryRecvError::Empty) => return (objs, false),
//            Err(mpsc::TryRecvError::Disconnected) => {
//                #[cfg(test)]
//                println!("exiting epoch-gc, disconnected");
//                return (objs, true);
//            }
//        }
//    }
//}
