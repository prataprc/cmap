use std::{
    borrow::Borrow,
    hash::Hash,
    sync::{
        atomic::{AtomicPtr, Ordering::SeqCst},
        mpsc,
    },
};

const SLOT_MASK: u64 = 0xff;

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
                    Some(entry) if istagged(entry) => continue 'retry,
                    Some(entry) => {
                        let entry = unsafe { entry.as_ref().unwrap() };
                        match entry.borrow_key::<Q>() {
                            Some(ekey) if ekey == key => {
                                break 'retry Some(entry.as_value().clone());
                            }
                            Some(_) => {
                                parent = entry;
                                node = entry.as_next_ptr();
                            }
                            None => break 'retry None,
                        }
                    }
                    None => unreachable!(),
                }
            }
        }
    }

    fn set(head: &Entry<K, V>, key: K, value: V, epoch: u64, tx: ReclaimTx<K, V>) -> Option<Box<V>>
    where
        K: PartialEq + Clone,
        V: Clone,
    {
        let mut value = Box::new(value);

        'retry: loop {
            // head must be entry list's first sentinel, skip it.
            let mut parent: &Entry<K, V> = head;
            let mut node: Option<*mut Entry<K, V>> = head.as_next_ptr(); // skip sentinel

            loop {
                match node {
                    Some(entry) if istagged(entry) => continue 'retry,
                    Some(entry) => {
                        let entry = unsafe { entry.as_ref().unwrap() };
                        match entry.borrow_key::<K>() {
                            Some(entry_key) if entry_key == &key => {
                                let old_value = entry.set_value(epoch, value, tx);
                                break 'retry Some(old_value);
                            }
                            Some(_) => {
                                parent = entry;
                                node = entry.as_next_ptr();
                            }
                            None => {
                                let next = match parent {
                                    Entry::E { next, .. } => next,
                                    Entry::S { .. } | Entry::N => unreachable!(),
                                };

                                let (key, old) = (key.clone(), node.unwrap());
                                let new = Box::leak(Entry::new(key, value, old));
                                if next.compare_and_swap(old, new, SeqCst) != old {
                                    value = unsafe { Box::from_raw(new).take_value() };
                                    continue 'retry;
                                } else {
                                    break 'retry None;
                                }
                            }
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
                    Some(entry) if istagged(entry) => continue 'retry,
                    Some(entry) => {
                        let entry = unsafe { entry.as_ref().unwrap() };
                        match entry.borrow_key::<Q>() {
                            Some(entry_key) if entry_key == key => match entry {
                                Entry::E { next, .. } => {
                                    // first CAS
                                    let old = next.load(SeqCst);
                                    let new = tag(old);
                                    if next.compare_and_swap(old, new, SeqCst) != old {
                                        continue 'retry;
                                    }
                                    // second CAS
                                    let new = untag(next.load(SeqCst));
                                    let next = match parent {
                                        Entry::E { next, .. } => next,
                                        Entry::S { .. } | Entry::N => unreachable!(),
                                    };
                                    let old = node.unwrap();
                                    if next.compare_and_swap(old, new, SeqCst) == old {
                                        let entry = unsafe { Box::from_raw(old) };
                                        let oval = Box::new(entry.as_value().clone());
                                        tx.send(Reclaim::Entry {
                                            epoch,
                                            entry: entry,
                                        });
                                        return Some(oval);
                                    } else {
                                        continue 'retry;
                                    }
                                }
                                Entry::S { .. } | Entry::N => unreachable!(),
                            },
                            Some(_) => {
                                parent = entry;
                                node = entry.as_next_ptr();
                            }
                            None => return None,
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

#[inline]
fn hamming_distance(x: u128, y: u128) -> usize {
    // TODO: optimize it with SSE or popcnt instructions, figure-out a way.
    (x ^ y).count_ones() as usize
}
