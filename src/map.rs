use std::{
    borrow::Borrow,
    hash::Hash,
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        mpsc, RwLock,
    },
};

use crate::gc::Gc;

const SLOT_MASK: u64 = 0xff;

pub struct Map<K, V, H> {
    id: usize,
    root: AtomicPtr<In<K, V>>,
    epoch: AtomicU64,
    access_log: Arc<RwLock<Vec<AtomicU64>>>,
    gc: Gc<K, V>,
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

impl<K, V> Node<K, V> -> Box<Node<K, V>>{
    fn to_node_list(
        &self, n: usize, entry: *mut Entry<K, V>
    ) -> (*mut Node<K, V>,  {
        match self {
            Node::Trie { bmp, childs, } => match childs[n].load(SeqCst).as_ref().unwrap() {
                Child::Leaf(leaf) => {
                    let leaf = unsafe { leaf as *mut Entry<K, V> };
                    let mut head = Entry::new_list();
                    head.insert(entry);
                    head.insert(leaf);

                    let bmp = bmp.clone();
                    let childs = childs.clone();

                    childs[n] = AtomicPtr::new(Box::leak(Box::new(In {
                        node: AtomicPtr::new(Box::new(Node::List { head }))
                    })));
                    Box::new(Node::Trie { bmp, childs })
                }
                Child::Deep(_) => unreachable!(),
            }
            Node::List { .. } => unreachable!(),
        };
    }

    fn leaf_match(&self, n: usize, key: &K) -> Option<bool> {
        match self {
            Node::Trie { childs, .. } => match childs[n].load(SeqCst).as_ref().unwrap() {
                Child::Deep(_) => None,
                Child::Leaf(leaf) if leaf.borrow_key() == key => Some(true),
                Child::Leaf(leaf) => Some(false),
            }
            Node::List { .. } => unreachable!(),
        }
    }

    fn get_trie(&self, n: usize) -> &In<K, V> {
        match self {
            Node::Trie { childs, .. } => match childs[n].load(SeqCst).as_ref().unwrap() {
                Child::Deep(inode) => inode,
                Child::Leaf(_) => unreachable!(),
            }
            Node::List { .. } => unreachable!(),
        }
    }

    fn set_list(
        &self, key: K, value: V, epoch: u64, gc: &Gc<K, V>
    ) -> Result<Option<Box<V>>> {
        match self {
            Node::List { head } => Entry::set(key, value, head, epoch, &self.gc)?,
            Node::Trie { .. } => unreachable!(),
        }
    }

    fn insert_trie(&self, w: u8, n: usize, child: *mut Child<K, V>) -> Box<Node<K, V>> {
        match self {
            Node::Trie { bmp, childs } => {
                let bmp = {
                    let mut bmp = bmp.clone();
                    if w < 128 {
                        bmp[0] = bmp[0] | (1_u128 << w);
                    } else {
                        bmp[1] = bmp[1] | (1_u128 << (w - 128));
                    };
                    bmp
                };
                let childs = childs.clone();
                childs.insert(n, AtomicPtr::new(child));
                Box::new(Node::Trie { bmp, childs })
            }
            Node::List { .. } => unreachable!(),
        }
    }

    fn update_leaf(&self, n: usize, val: V, epoch: u64, gc: &Gc<K, V>) -> Result<Box<V>> {
        match self {
            Node::Trie { childs, .. } => match childs[n].load(SeqCst).as_ref().unwrap() {
                Child::Leaf(leaf) => {
                    Ok(leaf.set_value(Box::new(val), epoch, gc)?)
                }
                Child::Deep(_) => unreachable!(),
            }
            Node::List { .. } => unreachable!(),
        }
    }

    fn leaf_to_list(&self, n: usize, entry: *mut Entry<K, V>) -> Box<Node<K, V>> {
        match self {
            Node::Trie { childs, bmp } => match childs[n].load(SeqCst).as_ref().unwrap() {
                Child::Leaf(leaf) => {
                    let mut head = Entry::new_list();
                    head.insert(entry);
                    head.insert(unsafe { leaf as *mut Entry<K, V> });

                    let node = AtomicPtr::new(Box::leak(Box::new(Node::List { head })));
                    Child::Deep(In { node })
                }
                Child::Deep(_) => unreachable!(),
            }
            Node::List { .. } => unreachable!(),
        }
    }

    fn hamming_distance(&self, w: u8) -> Distance {
        match self {
            Node::Trie { bmp, .. } => hamming_distance(w, bmp.clone()),
            Node::List { .. } => unreachable!(),
        }
    }
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
        let access_log = vec![ AtomicU64::new(1) ];
        Map {
            id: 0,
            root: AtomicPtr::new(Box::leak(root)),
            epoch: AtomicU64::new(1),
            access_log: RwLock::new(access_log);
        }
    }

    fn register_epoch(&self) {
        let epoch = self.epoch.fetch_add(1);
        let log = self.access_log.read().unwrap();
        log[self.id].store(epoch);
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

        let mut inode: &In<K, V> = self.root.load(SeqCst).as_ref().unwrap();

        let value = loop {
            let node = inode.node.load(SeqCst).as_ref().unwrap();
            inode = match (node, ws.pop()) {
                (Node::Trie { bmp, childs }, Some(w)) => {
                    let dist = hamming_distance(w, bmp.clone());
                    let child = match dist {
                        Distance::Set(n) => childs[n].load(SeqCst).as_ref().unwrap(),
                        Distance::Insert(n) => break None,
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

        self.register_epoch();

        value
    }

    pub fn set<Q>(&self, key: K, value: V) -> Result<Option<Box<V>>>
    where
        K: PartialEq + Clone + Hash,
        V: Clone,
    {
        let ws = key_to_hashbits(key);

        'retry loop {
            let epoch = self.epoch.load(SeqCst);
            let mut inode: &In<K, V> = self.root.load(SeqCst).as_ref().unwrap();

            loop {
                let old = inode.node.load(SeqCst);
                let node = old.as_ref().unwrap();

                let w = match ws.pop() {
                    None => break 'retry node.set_list(key, value, epoch, &self.gc),
                    Some(w) => w,
                };

                let n = match node.hamming_distance(w) {
                    Distance::Insert(n) => {
                        let child = {
                            let value = Box::new(value.clone());
                            let child = Child::Leaf(Entry::new_leaf(key.clone(), value));
                            Box::new(child)
                        };
                        let child_ptr = Box::leak(child);
                        let new = Box::leak(node.insert_trie(w, n, child_ptr));

                        if inode.node.compare_and_swap(old, new, SeqCst) == old {
                            let node = Box::from_raw(old);
                            err_at!(GcFail, self.gc.send(Reclaim::Node { epoch, node }))?;
                            break 'retry Ok(None)
                        } else {
                            let _child = Box::from_raw(child_ptr);
                            let _new = Box::from_raw(new);
                            continue 'retry
                        }
                    }
                    Distance::Set(n) => n,
                };

                let leaf_match = node.leaf_match(n);
                match leaf_match {
                    Some(true) => break 'retry node.update_leaf(n, value, epoch, &self.gc),
                    Some(false) if ws.len() == 0 => {
                        // reached tip of the tree convert leaf to list.
                        let entry_ptr = {
                            let value = Box::new(value.clone());
                            let entry = Entry::new_leaf(key.clone(), value)
                            let entry_ptr = Box::leak(Box::new(entry));
                        };
                        Node::new_list(entry_ptr)

                        let new = Box::leak(node.leaf_to_list(n, entry));

                        if inode.node.compare_and_swap(old, new, SeqCst) == old {
                            let node = Box::from_raw(old);
                            err_at!(GcFail, self.gc.send(Reclaim::Node { epoch, node }))?;
                            break 'retry Ok(None)
                        } else {
                            continue 'retry
                        }
                    }
                    Some(false) => {
                        // create another level deeper.
                        todo!()
                    }
                    None => node.get_trie(n)
                }
            }
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

enum Distance {
    Set(usize),    // found
    Insert(usize), // not found
}

#[inline]
fn hamming_distance(w: u8, bmp: [u128; 2]) -> Distance {
    let posn = 1 << w;
    let mask = !(posn - 1);
    let bmp: u128 = if w < 128 { bmp[0] } else { bmp[1] };

    let (x, y) = ((bmp & mask), bmp);
    // TODO: optimize it with SSE or popcnt instructions, figure-out a way.
    let dist = (x ^ y).count_ones() as usize;

    match (bmp & posn) {
        0 => Distance::Insert(dist),
        _ => Distance::Set(dist),
    }
}

// TODO: Can we make this to use a generic hash function ?
#[inline]
fn key_to_hashbits<Q>(key: &Q) -> Vec<u8>
where
    Q: Hash,
{
    use fasthash::city;

    let mut hasher = city::Hash32;
    key.hash(&mut hasher);
    let code: u64 = hasher.finish();

    let mut ws: Vec<u8> = (0..8).map(|i| ((code >> (i * 8)) && 0xFF) as u8).collect();
    ws.reverse();
}
