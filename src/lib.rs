use std::{
    borrow::Borrow,
    sync::{
        atomic::{AtomicPtr, Ordering::SeqCst},
        Arc,
    },
};

/// This is total magic !!!

enum Entry<K, V> {
    S {
        next: AtomicPtr<Arc<Entry<K, V>>>,
    },
    E {
        key: K,
        value: AtomicPtr<Arc<V>>,
        next: AtomicPtr<Arc<Entry<K, V>>>,
    },
    N,
}

impl<K, V> Entry<K, V> {
    fn new() -> Box<Arc<Entry<K, V>>> {
        let tail = {
            let entry = Entry::S {
                next: AtomicPtr::new(Box::leak(Box::new(Arc::new(Entry::N)))),
            };
            Box::new(Arc::new(entry))
        };

        let head = Entry::S {
            next: AtomicPtr::new(Box::leak(tail)),
        };

        Box::new(Arc::new(head))
    }

    fn new_entry(key: K, value: Box<Arc<V>>, next: Box<Arc<Entry<K, V>>>) -> Box<Arc<Entry<K, V>>> {
        let entry = Entry::E {
            key,
            value: AtomicPtr::new(Box::leak(value)),
            next: AtomicPtr::new(Box::leak(next)),
        };

        Box::new(Arc::new(entry))
    }

    fn drop_fields(&self) {
        match self {
            Entry::S { next } => unsafe {
                let ptr = next.load(SeqCst).as_ref().unwrap();
                let _next = Arc::from_raw(Arc::as_ptr(ptr));
            },
            Entry::E { value, next, .. } => unsafe {
                let ptr = next.load(SeqCst).as_ref().unwrap();
                let _next = Arc::from_raw(Arc::as_ptr(ptr));
                let ptr = value.load(SeqCst).as_ref().unwrap();
                let _value = Arc::from_raw(Arc::as_ptr(ptr));
            },
            Entry::N => (),
        }
    }
}

impl<K, V> Entry<K, V> {
    fn len(entry: Arc<Entry<K, V>>) -> usize {
        let mut count = 0;
        let mut node = entry.to_next().unwrap();

        while let Some(n) = node.to_next() {
            node = n;
            count += 1;
        }

        count
    }

    fn get<Q>(entry: Arc<Entry<K, V>>, key: &Q) -> Option<Arc<V>>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        let mut node: Option<Arc<Entry<K, V>>> = entry.to_next(); // skip sentinel

        loop {
            node = match node {
                Some(entry) => match entry.borrow_key::<Q>() {
                    Some(entry_key) if entry_key == key => break Some(entry.to_value()),
                    _ => entry.to_next(),
                },
                None => break None,
            }
        }
    }

    fn set(entry: Arc<Entry<K, V>>, key: K, value: V) -> Option<Arc<V>>
    where
        K: PartialEq + Clone,
        V: Clone,
    {
        'retry: loop {
            let mut node: Option<Arc<Entry<K, V>>> = entry.to_next(); // skip sentinel
            let mut parent = Arc::clone(&entry);

            loop {
                match node {
                    Some(entr) => match entr.borrow_key::<K>() {
                        Some(entry_key) if entry_key == &key => {
                            let value = Box::new(Arc::new(value));
                            let old_value = entr.set_value(value);
                            return Some(old_value);
                        }
                        Some(_) => {
                            node = entr.to_next();
                            parent = entr;
                        }
                        None => match parent.as_ref() {
                            Entry::E { next, .. } => {
                                let old = untag(next.load(SeqCst));
                                let entr = {
                                    let value = Box::new(Arc::new(value.clone()));
                                    Entry::new_entry(key.clone(), value, Box::new(entr))
                                };
                                let new = Box::leak(entr);
                                if next.compare_and_swap(old, new, SeqCst) == old {
                                    return None;
                                } else {
                                    let entr = unsafe { Box::from_raw(new) };
                                    entr.drop_fields();
                                    continue 'retry;
                                }
                            }
                            Entry::S { .. } | Entry::N => unreachable!(),
                        },
                    },
                    None => unreachable!(),
                }
            }
        }
    }

    fn remove<Q>(entry: Arc<Entry<K, V>>, key: &Q) -> Option<Arc<V>>
    where
        K: Borrow<Q>,
        V: Clone,
        Q: PartialEq + ?Sized,
    {
        'retry: loop {
            let mut node: Option<Arc<Entry<K, V>>> = entry.to_next(); // skip sentinel
            let mut parent = Arc::clone(&entry);

            loop {
                match node {
                    Some(entr) => match entr.borrow_key::<Q>() {
                        Some(entry_key) if entry_key == key => match entr.as_ref() {
                            Entry::E { next, .. } => {
                                // double CAS
                                let old = untag(next.load(SeqCst));
                                let new = tag(old);
                                // first CAS
                                if next.compare_and_swap(old, new, SeqCst) != old {
                                    continue 'retry;
                                }
                                let new = untag(next.load(SeqCst));
                                let old_value = Some(entr.to_value());
                                // second CAS
                                match parent.as_ref() {
                                    Entry::E { next, .. } => {
                                        let old = untag(next.load(SeqCst));
                                        if next.compare_and_swap(old, new, SeqCst) == old {
                                            entr.drop_fields();
                                            return old_value;
                                        } else {
                                            continue 'retry;
                                        }
                                    }
                                    Entry::S { .. } | Entry::N => unreachable!(),
                                }
                            }
                            Entry::S { .. } | Entry::N => unreachable!(),
                        },
                        Some(_) => {
                            node = entr.to_next();
                            parent = entr;
                        }
                        None => return None,
                    },
                    None => unreachable!(),
                }
            }
        }
    }
}

impl<K, V> Entry<K, V> {
    fn set_value(&self, newval: Box<Arc<V>>) -> Arc<V> {
        match self {
            Entry::E { value, .. } => {
                let old_value = unsafe {
                    let v = value.load(SeqCst).as_ref().unwrap();
                    Arc::from_raw(Arc::as_ptr(v))
                };
                value.store(Box::leak(newval), SeqCst);
                old_value
            }
            Entry::S { .. } | Entry::N => unreachable!(),
        }
    }

    fn set_next(&self, entry: Box<Arc<Entry<K, V>>>) {
        match self {
            Entry::S { next, .. } | Entry::E { next, .. } => {
                next.store(Box::leak(entry), SeqCst);
            }
            Entry::N => unreachable!(),
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

    fn to_value(&self) -> Arc<V> {
        match self {
            Entry::E { value, .. } => {
                let valref = unsafe { value.load(SeqCst).as_ref().unwrap() };
                Arc::clone(valref)
            }
            _ => unreachable!(),
        }
    }

    fn to_next(&self) -> Option<Arc<Entry<K, V>>> {
        match self {
            Entry::S { next } | Entry::E { next, .. } => {
                let ptr = next.load(SeqCst);
                unsafe { Some(Arc::clone(untag(ptr).as_ref().unwrap())) }
            }
            Entry::N => None,
        }
    }

    fn to_next_location(&self) -> &AtomicPtr<Arc<Entry<K, V>>> {
        match self {
            Entry::S { next } | Entry::E { next, .. } => next,
            Entry::N => unreachable!(),
        }
    }

    fn tag(&self) {
        loop {
            match self {
                Entry::S { next } | Entry::E { next, .. } => {
                    let ptr = next.load(SeqCst);
                    let newptr = tag(ptr);
                    if next.compare_and_swap(ptr, newptr, SeqCst) == ptr {
                        break;
                    }
                }
                Entry::N => break,
            }
        }
    }

    fn untag(&self) {
        loop {
            match self {
                Entry::S { next } | Entry::E { next, .. } => {
                    let ptr = next.load(SeqCst);
                    let newptr = untag(ptr);
                    if next.compare_and_swap(ptr, newptr, SeqCst) == ptr {
                        break;
                    }
                }
                Entry::N => break,
            }
        }
    }

    fn istagged(&self) -> Option<bool> {
        match self {
            Entry::S { next } | Entry::E { next, .. } => Some(istagged(next.load(SeqCst))),
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
