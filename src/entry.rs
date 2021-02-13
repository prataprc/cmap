use std::{
    borrow::Borrow,
    sync::atomic::{AtomicPtr, Ordering::SeqCst},
};

use crate::{
    gc::{Gc, Reclaim},
    Error, Result,
};

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
        Box::new(Entry::E { key, value, next })
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

    pub fn set(
        nkey: K,
        nvalue: V,
        head: &Entry<K, V>,
        epoch: u64,
        gc: &Gc<K, V>,
    ) -> Result<Option<Box<V>>>
    where
        K: PartialEq + Clone,
        V: Clone,
    {
        let swing = |parent: &Entry<K, V>, old: *mut Entry<K, V>| -> bool {
            let next = match parent {
                Entry::E { next, .. } => next,
                Entry::S { next } => next,
                Entry::N => unreachable!(),
            };

            let new = Box::leak(Entry::new(key.clone(), value.clone(), old));
            if next.compare_and_swap(old, new, SeqCst) == old {
                return true;
            } else {
                unsafe { Box::from_raw(new) };
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
                        let cas = gc::Cas::new(epoch, &gc);

                        match unsafe { node_ptr.as_ref().unwrap() } {
                            Entry::E { key, value, next } if key == nkey => {
                                let new = {
                                    let next = AtomicPtr::new(next.load(SeqCst));
                                    let entry = Entry::E { key, value: nvalue, next };
                                    Box::leak(Box::new(entry))
                                };
                                cas.free_on_pass(gc::Mem::Entry(node_ptr));
                                cas.free_on_fail(gc::Mem::Entry(new));
                                if cas.swing(parent.as_atomicptr(), node_ptr, new) {
                                    break 'retry Ok(Some(value.clone()),
                                } else {
                                    continue 'retry
                                }
                            }
                            node_ref @ Entry::E { .. } => {
                                parent = node_ref;
                                node = node_ref.as_next_ptr();
                            }
                            Entry::S { .. } => {
                                let new = {
                                    let next = AtomicPtr::new(node_ptr);
                                    let entry = Entry::E { key, value: nvalue, next };
                                    Box::leak(Box::new(entry))
                                };
                                cas.free_on_fail(gc::Mem::Entry(new));
                                if cas.swing(parent.as_atomicptr(), node_ptr, new) {
                                    break 'retry Ok(None)
                                } else {
                                    continue 'retry
                                }
                            }
                        }
                    }
                    None => unreachable!(),
                }
            }
        }
    }

    pub fn remove<Q>(
        key: &Q,
        head: &Entry<K, V>,
        epoch: u64,
        gc: &Gc<K, V>,
    ) -> Result<Option<Box<V>>>
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
                            None => break 'retry Ok(None),
                        };

                        let cas = gc::Cas::new(epoch, &gc);

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
                                    let (old, new) = (node_ptr, next_ptr);
                                    if next.compare_and_swap(old, new, SeqCst) == old {
                                        let entry = unsafe { Box::from_raw(old) };
                                        let oval = Box::new(entry.as_value().clone());
                                        let rclm = Reclaim::Entry { epoch, entry };
                                        err_at!(GcFail, gc.post(rclm))?;
                                        break 'retry Ok(Some(oval));
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
                            None => break 'retry Ok(None),
                        }
                    }
                    None => unreachable!(),
                }
            }
        }
    }
}

impl<K, V> Entry<K, V> {
    fn set_value(&self, nval: Box<V>, epoch: u64, gc: &Gc<K, V>) -> Result<Box<V>>
    where
        V: Clone,
    {
        match self {
            Entry::E { value, .. } => {
                let old_value = unsafe { Box::from_raw(value.load(SeqCst)) };
                let rclm = Reclaim::Value {
                    epoch,
                    value: old_value.clone(),
                };
                err_at!(GcFail, gc.post(rclm))?;
                value.store(Box::leak(nval), SeqCst);
                Ok(old_value)
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
