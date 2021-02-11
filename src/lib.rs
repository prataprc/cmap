use fasthash::city;

use std::{
    borrow::Borrow,
    hash::Hash,
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        mpsc, RwLock,
    },
};

const SLOT_MASK: u64 = 0xff;

pub struct Map<K, V, H> {
    id: usize,
    root: AtomicPtr<In<K, V>>,
    epoch: AtomicU64,
    access_log: RwLock<Vec<AtomicU64>>,
}

struct In<K, V> {
    node: AtomicPtr<Node<K, V>>,
}

enum Node<K, V> {
    Trie {
        bmp: [u128; 2],
        childs: Vec<AtomicPtr<Child<K, V>>>,
    },
    List {
        head: AtomicPtr<Entry<K, V>>,
    },
}

enum Child<K, V> {
    Deep(In<K, V>),
    Leaf(Entry<K, V>),
}

impl<K, V> Map<K, V> {
    pub fn new() -> Map<K, V> {
        let node = Box::new(Node::Trie {
            bmp: [0_u128; 2],
            childs: Vec::default(),
        });
        let root = Box::new(In {
            node: AtomicPtr::new(Box::leak(node)),
        });
        Map {
            id: 0,
            root: AtomicPtr::new(Box::leak(root)),
            epoch: AtomicU64::new(1),
            access_log: RwLock::new(Vec::default()),
        }
    }

    fn register_epoch(&self) {
        let epoch = self.epoch.fetch_add(1);
        let mut log = self.access_log.read().unwrap();
        access[self.off].store(epoch);
    }
}

impl<K, V> Map<K, V> {
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized + Hash,
    {
        let ws = key_to_hashbits(key);

        let _epoch = self.epoch.load(SeqCst);
        let mut inode: &In<K, V> = self.root.load(SeqCst).as_ref().unwrap();

        let value = loop {
            inode = match (inode.node.load(SeqCst).as_ref().unwrap(), ws.pop()) {
                (Node::Trie { bmp, childs }, Some(w)) => {
                    let dist = hamming_distance(w, bmp.clone());
                    let child = match dist {
                        HammDistance::Set(n) => childs[n].load(SeqCst).as_ref().unwrap(),
                        HammDistance::Insert(n) => break None,
                    };
                    match child {
                        Child::Deep(inode) => inode,
                        Child::Leaf(entry) if entry.borrow_key::<Q>() == key => {
                            break Some(entry.as_value().clone());
                        }
                        Child::Leaf(entry) => break None,
                    }
                }
                (Node::Trie { .. }, None) => unreachable!(),
                (Node::List { head }, _) => break Entry::get(head, key),
            }
        };

        self.register_epoch()

        value
    }

    pub fn set<Q>(&self, key: K, value: V) -> Option<Box<V>>
    where
        K: PartialEq + Clone + Hash,
        V: Clone,
    {
        let ws = key_to_hashbits(key);
        let epoch = self.epoch.load(SeqCst);

        let inode = self.root.load(SeqCst).as_ref().unwrap();
        let node = inode.node.load(SeqCst).as_ref().unwrap();

        match node {
            Node::Trie { bmp, childs } => {
                let w = ws.pop();
                match hamming_distance(w, bmp.clone()) {
                    HammDistance::Set(n) if ws.len() == 0 => (),
                    HammDistance::Set(n) => (),
                    HammDistance::Insert(n) if ws.len() == 0 => (),
                    HammDistance::Insert(n) => (),
                }
            }
            Node::List { head } => {
                let head = head.load(SeqCst).as_ref().unwrap();
                Entry::set(head, epoch, tx, key, value)
            }
            Node::Tomb { item } => todo!(),
        }

        self.register_epoch()
    }

    pub fn remove<Q>(&self, key: &Q) -> Option<Box<V>>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        todo!()
    }
}

impl<K, V> In<K, V> {
    fn lookup<Q>(inode: &In<K, V>, key: &Q) -> Option<V> {}
}

type ReclaimTx<K, V> = mpsc::Sender<Reclaim<K, V>>;

enum Reclaim<K, V> {
    Value { epoch: u64, value: Box<V> },
    Entry { epoch: u64, entry: Box<Entry<K, V>> },
}

enum Entry<K, V> {
    S {
        next: AtomicPtr<Entry<K, V>>,
    },
    E {
        key: K,
        value: AtomicPtr<V>,
        next: AtomicPtr<Entry<K, V>>,
    },
    N,
}

impl<K, V> Entry<K, V> {
    fn new_list() -> Box<Entry<K, V>> {
        // create 2 sentinels, wire them up and return.
        let tail = {
            let entry = Entry::S {
                next: AtomicPtr::new(Box::leak(Box::new(Entry::N))),
            };
            Box::new(entry)
        };

        let head = Entry::S {
            next: AtomicPtr::new(Box::leak(tail)),
        };

        Box::new(head)
    }

    fn new(key: K, value: Box<V>, next: *mut Entry<K, V>) -> Box<Entry<K, V>> {
        let value = AtomicPtr::new(Box::leak(value));
        let next = AtomicPtr::new(next);
        Box::new(Entry::E { key, value, next })
    }
}

impl<K, V> Entry<K, V> {
    fn get<Q>(head: &Entry<K, V>, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        'retry: loop {
            // head must be entry list's first sentinel, skip it.
            let mut parent: &Entry<K, V> = head;
            let mut node: Option<*mut Entry<K, V>> = head.as_next_ptr();

            loop {
                match node {
                    Some(node_ptr) if istagged(node_ptr) => continue 'retry,
                    Some(node_ptr) => {
                        let node_ref = unsafe { node_ptr.as_ref().unwrap() };
                        match node_ref.borrow_key::<Q>() {
                            Some(ekey) if ekey == key => {
                                break 'retry Some(node_ref.as_value().clone());
                            }
                            Some(_) => {
                                parent = node_ref;
                                node = node_ref.as_next_ptr();
                            }
                            None => break 'retry None,
                        }
                    }
                    None => unreachable!(),
                }
            }
        }
    }

    fn set(head: &Entry<K, V>, epoch: u64, tx: ReclaimTx<K, V>, key: K, value: V) -> Option<Box<V>>
    where
        K: PartialEq + Clone,
        V: Clone,
    {
        let mut value = Box::new(value);

        let swing = |parent: &Entry<K, V>, old: *mut Entry<K, V>| -> bool {
            let next = match parent {
                Entry::E { next, .. } => next,
                Entry::S { .. } | Entry::N => unreachable!(),
            };

            let new = Box::leak(Entry::new(key.clone(), value.clone(), old));
            if next.compare_and_swap(old, new, SeqCst) == old {
                return true;
            } else {
                let _value = unsafe { Box::from_raw(new).take_value() };
                return false;
            }
        };

        'retry: loop {
            // head must be entry list's first sentinel, skip it.
            let mut parent: &Entry<K, V> = head;
            let mut node: Option<*mut Entry<K, V>> = head.as_next_ptr(); // skip sentinel

            loop {
                match node {
                    Some(node_ptr) if istagged(node_ptr) => continue 'retry,
                    Some(node_ptr) => {
                        let node_ref = unsafe { node_ptr.as_ref().unwrap() };
                        match node_ref.borrow_key::<K>() {
                            Some(entry_key) if entry_key == &key => {
                                let old_value = node_ref.set_value(epoch, value, tx);
                                break 'retry Some(old_value);
                            }
                            Some(_) => {
                                parent = node_ref;
                                node = node_ref.as_next_ptr();
                            }
                            None if swing(parent, node.unwrap()) => break 'retry None,
                            None => continue 'retry,
                        }
                    }
                    None => unreachable!(),
                }
            }
        }
    }

    fn remove<Q>(head: &Entry<K, V>, key: &Q, epoch: u64, tx: ReclaimTx<K, V>) -> Option<Box<V>>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        'retry: loop {
            // head must be entry list's first sentinel, skip it.
            let mut parent: &Entry<K, V> = head;
            let mut node: Option<*mut Entry<K, V>> = head.as_next_ptr(); // skip sentinel

            loop {
                match node {
                    Some(node_ptr) if istagged(node_ptr) => continue 'retry,
                    Some(node_ptr) => {
                        let node_ref = unsafe { node_ptr.as_ref().unwrap() };
                        let next_ptr = match node_ref.as_next_ptr() {
                            Some(next_ptr) if istagged(next_ptr) => continue 'retry,
                            Some(next_ptr) => next_ptr,
                            None => break 'retry None,
                        };
                        match node_ref.borrow_key::<Q>() {
                            Some(entry_key) if entry_key == key => match node_ref {
                                Entry::E { next, .. } => {
                                    // first CAS
                                    let (old, new) = (next_ptr, tag(next_ptr));
                                    if next.compare_and_swap(old, new, SeqCst) != old {
                                        continue 'retry;
                                    }
                                    // second CAS
                                    let next = match parent {
                                        Entry::E { next, .. } => next,
                                        Entry::S { .. } | Entry::N => unreachable!(),
                                    };
                                    let (old, new) = (node.unwrap(), next_ptr);
                                    if next.compare_and_swap(old, new, SeqCst) == old {
                                        let entry = unsafe { Box::from_raw(old) };
                                        let oval = Box::new(entry.as_value().clone());
                                        tx.send(Reclaim::Entry { epoch, entry });
                                        break 'retry Some(oval);
                                    } else {
                                        continue 'retry;
                                    }
                                }
                                Entry::S { .. } | Entry::N => unreachable!(),
                            },
                            Some(_) => {
                                parent = node_ref;
                                node = node_ref.as_next_ptr();
                            }
                            None => break 'retry None,
                        }
                    }
                    None => unreachable!(),
                }
            }
        }
    }
}

impl<K, V> Entry<K, V> {
    fn set_value(&self, epoch: u64, nval: Box<V>, chan: ReclaimTx<K, V>) -> Box<V>
    where
        V: Clone,
    {
        match self {
            Entry::E { value, .. } => {
                let old_value = unsafe { Box::from_raw(value.load(SeqCst)) };
                chan.send(Reclaim::Value {
                    epoch,
                    value: old_value.clone(),
                });
                value.store(Box::leak(nval), SeqCst);
                old_value
            }
            Entry::S { .. } | Entry::N => unreachable!(),
        }
    }

    fn borrow_key<Q>(&self) -> Option<&Q>
    where
        K: Borrow<Q>,
        Q: ?Sized,
    {
        match self {
            Entry::E { key, .. } => Some(key.borrow()),
            Entry::S { .. } => None,
            Entry::N => unreachable!(),
        }
    }

    fn as_value(&self) -> &V {
        match self {
            Entry::E { value, .. } => unsafe { value.load(SeqCst).as_ref().unwrap() },
            _ => unreachable!(),
        }
    }

    fn take_value(&self) -> Box<V> {
        match self {
            Entry::E { value, .. } => unsafe { Box::from_raw(value.load(SeqCst)) },
            _ => unreachable!(),
        }
    }

    fn as_next_ptr(&self) -> Option<*mut Entry<K, V>> {
        match self {
            Entry::S { next } | Entry::E { next, .. } => Some(next.load(SeqCst)),
            Entry::N => None,
        }
    }
}

fn tag<T>(ptr: *mut T) -> *mut T {
    let ptr = ptr as u64;
    assert!(ptr & 0x1 == 0);
    (ptr | 0x1) as *mut T
}

fn untag<T>(ptr: *mut T) -> *mut T {
    let ptr = ptr as u64;
    (ptr & !1) as *mut T
}

fn istagged<T>(ptr: *mut T) -> bool {
    let ptr = ptr as u64;
    (ptr & 0x1) == 1
}

enum HammDistance {
    Insert(usize),
    Set(usize),
}

#[inline]
fn hamming_distance(w: u8, bmp: [u128; 2]) -> HammDistance {
    let posn = 1 << w;
    let mask = !(posn - 1);
    let bmp: u128 = if w < 128 { bmp[0] } else { bmp[1] };

    let (x, y) = ((bmp & mask), bmp);
    // TODO: optimize it with SSE or popcnt instructions, figure-out a way.
    let dist = (x ^ y).count_ones() as usize;

    match (bmp & posn) {
        0 => HammDistance::Insert(dist),
        _ => HammDistance::Set(dist),
    }
}

#[inline]
fn key_to_hashbits<Q>(key: &Q) -> Vec<u8>
where
    Q: Hash,
{
    let mut hasher = city::Hash32;
    key.hash(&mut hasher);
    let code: u64 = hasher.finish();

    let mut ws: Vec<u8> = (0..8).map(|i| ((code >> (i * 8)) && 0xFF) as u8).collect();
    ws.reverse();
}
