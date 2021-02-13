use std::{
    borrow::Borrow,
    sync::atomic::{AtomicPtr, Ordering::SeqCst},
};

use crate::gc::{self, Gc};

pub enum Entry<K, V> {
    S {
        next: AtomicPtr<Entry<K, V>>,
    },
    E {
        key: K,
        value: V,
        next: AtomicPtr<Entry<K, V>>,
    },
    N,
}

impl<K, V> Entry<K, V> {
    pub fn new(key: K, value: V, next: *mut Entry<K, V>) -> Box<Entry<K, V>> {
        let next = AtomicPtr::new(next);
        Box::new(Entry::E { key, value, next })
    }

    pub fn new_list() -> Box<Entry<K, V>> {
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

    pub fn new_leaf(key: K, value: V) -> Entry<K, V> {
        let next = AtomicPtr::new(Box::leak(Box::new(Entry::N)));
        Entry::E { key, value, next }
    }

    pub fn insert(&mut self, entry: *mut Entry<K, V>) {
        match self {
            Entry::S { next } => next.store(entry, SeqCst),
            Entry::E { .. } | Entry::N => unreachable!(),
        }
    }
}

impl<K, V> Entry<K, V> {
    pub fn get<Q>(head: &Entry<K, V>, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        'retry: loop {
            // head must be entry list's first sentinel, skip it.
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
                            Some(_) => node = node_ref.as_next_ptr(),
                            None => break 'retry None,
                        }
                    }
                    None => unreachable!(),
                }
            }
        }
    }

    pub fn set(&self, nkey: K, nvalue: V, epoch: u64, gc: &Gc<K, V>) -> Option<V>
    where
        K: PartialEq + Clone,
        V: Clone,
    {
        'retry: loop {
            // head must be entry list's first sentinel, skip it.
            let mut parent: &Entry<K, V> = self;
            let (mut node_ptr, mut next_ptr) = get_pointers(parent);

            loop {
                if istagged(next_ptr) {
                    continue 'retry;
                }

                match unsafe { node_ptr.as_ref().unwrap() } {
                    Entry::E { key, value, next } if key == &nkey => {
                        let mut cas = gc::Cas::new(epoch, &gc);
                        let new = {
                            let next = AtomicPtr::new(next.load(SeqCst));
                            let (key, value) = (nkey.clone(), nvalue.clone());
                            let entry = Entry::E { key, value, next };
                            Box::leak(Box::new(entry))
                        };
                        cas.free_on_pass(gc::Mem::Entry(node_ptr));
                        cas.free_on_fail(gc::Mem::Entry(new));
                        if cas.swing(parent.as_atomicptr(), node_ptr, new) {
                            break 'retry Some(value.clone());
                        } else {
                            continue 'retry;
                        }
                    }
                    node_ref @ Entry::E { .. } => {
                        parent = node_ref;
                        (node_ptr, next_ptr) = get_pointers(parent);
                    }
                    Entry::S { .. } => {
                        let mut cas = gc::Cas::new(epoch, &gc);
                        let new = {
                            let next = AtomicPtr::new(node_ptr);
                            let (key, value) = (nkey.clone(), nvalue.clone());
                            let entry = Entry::E { key, value, next };
                            Box::leak(Box::new(entry))
                        };
                        cas.free_on_fail(gc::Mem::Entry(new));
                        if cas.swing(parent.as_atomicptr(), node_ptr, new) {
                            break 'retry None;
                        } else {
                            continue 'retry;
                        }
                    }
                    Entry::N => unreachable!(),
                }
            }
        }
    }

    pub fn remove<Q>(&self, dkey: &Q, epoch: u64, gc: &Gc<K, V>) -> Option<V>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        'retry: loop {
            // head must be entry list's first sentinel, skip it.
            let mut parent: &Entry<K, V> = self;
            let (mut node_ptr, mut next_ptr) = get_pointers(parent);

            loop {
                if istagged(next_ptr) {
                    continue 'retry;
                }

                match unsafe { node_ptr.as_ref().unwrap() } {
                    Entry::E { key, value, next } if key.borrow() == dkey => {
                        let mut cas = gc::Cas::new(epoch, &gc);
                        // first CAS
                        let (old, new) = (next_ptr, tag(next_ptr));
                        if next.compare_and_swap(old, new, SeqCst) != old {
                            continue 'retry;
                        }
                        // second CAS
                        cas.free_on_pass(gc::Mem::Entry(node_ptr));
                        if cas.swing(parent.as_atomicptr(), node_ptr, next_ptr) {
                            break 'retry Some(value.clone());
                        } else {
                            continue 'retry;
                        }
                    }
                    node_ref @ Entry::E { .. } => {
                        parent = node_ref;
                        (node_ptr, next_ptr) = get_pointers(parent);
                    }
                    Entry::S { .. } | Entry::N => break 'retry None,
                }
            }
        }
    }
}

impl<K, V> Entry<K, V> {
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
            Entry::E { value, .. } => value,
            _ => unreachable!(),
        }
    }

    fn as_next_ptr(&self) -> Option<*mut Entry<K, V>> {
        match self {
            Entry::S { next } | Entry::E { next, .. } => Some(next.load(SeqCst)),
            Entry::N => None,
        }
    }

    fn as_atomicptr(&self) -> &AtomicPtr<Entry<K, V>> {
        match self {
            Entry::S { next } | Entry::E { next, .. } => next,
            Entry::N => unreachable!(),
        }
    }
}

fn get_pointers<K, V>(parent: &Entry<K, V>) -> (*mut Entry<K, V>, *mut Entry<K, V>) {
    let node_ptr: *mut Entry<K, V> = parent.as_next_ptr().unwrap();
    let next_ptr = unsafe { node_ptr.as_ref().unwrap().as_next_ptr().unwrap() };
    (node_ptr, next_ptr)
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
