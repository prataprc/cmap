use mkit::thread;

use std::{
    sync::{
        atomic::{AtomicU64, Ordering::SeqCst},
        mpsc, Arc, RwLock,
    },
    thread::sleep,
    time::Duration,
};

use crate::{entry::Entry, Result};

// CAS operation

pub enum Mem<K, V> {
    Entry(*mut Entry<K, V>),
    Node(*mut Node<K, V>),
    Child(*mut Child<K, V>),
}

pub struct Cas<K, V, 'a> {
    epoch: u64,
    gc: &'a, Gc<K, V>
    pass: Vec<Mem<K, V>>
    fail: Vec<Mem<K, V>>
}

impl<K, V> Cas<K, V> {
    pub fn new(epoch: u64, gc: &Gc<K, V>) -> Cas<K, V> {
        Cas {
            epoch,
            gc,
            pass: Vec::default(),
            fail: Vec::default(),
        }
    }

    pub fn free_on_pass(&mut self, m: Mem<K, V>) {
        self.pass.push(m)
    }

    pub fn free_on_fail(&mut self, m: Mem<K, V>) {
        self.fail.push(m)
    }

    pub fn swing<T>(
        &mut self, loc: &AtomicPtr<T>, old: *mut T, new: *mut T
    ) -> Result<bool> {
        if loc.compare_and_swap(old, new, SeqCst) == old {
            for m in self.pass.drain(..) {
                match m {
                    Mem::Entry(ptr) => {
                        let entry = unsafe { Box::from_raw(ptr) };
                        let rclm = Reclaim::Entry { self.epoch, entry };
                        err_at!(GcFail, self.gc.send(rclm))?,
                    }
                    Mem::Node(ptr) => {
                        let _node = unsafe { Box::from_raw(ptr) };
                    }
                    Mem::Child(ptr) => {
                        let _child = unsafe { Box::from_raw(ptr) };
                    }
                }
            }
            true
        } else {
            for m in self.fail.drain(..) {
                match m {
                    Mem::Entry(ptr) => {
                        let _entry = unsafe { Box::from_raw(ptr) };
                    }
                    Mem::Node(ptr) => {
                        let _node = unsafe { Box::from_raw(ptr) };
                    }
                    Mem::Child(ptr) => {
                        let _child = unsafe { Box::from_raw(ptr) };
                    }
                }
            }
            false
        }
    }
}

pub enum Reclaim<K, V> {
    Value { epoch: u64, value: Box<V> },
    Entry { epoch: u64, entry: Box<Entry<K, V>> },
    // Node { epoch: u64, node: Box<Node<K, V>> },
}

pub type Gc<K, V> = thread::Tx<Reclaim<K, V>>;

pub struct GcThread<K, V> {
    access_log: Arc<RwLock<Vec<AtomicU64>>>,
    rx: thread::Rx<Reclaim<K, V>>,
}

impl<K, V> GcThread<K, V> {
    fn new(
        access_log: Arc<RwLock<Vec<AtomicU64>>>,
        rx: thread::Rx<Reclaim<K, V>>,
    ) -> GcThread<K, V> {
        GcThread { access_log, rx }
    }
}

impl<K, V> FnOnce<()> for GcThread<K, V> {
    type Output = Result<()>;

    extern "rust-call" fn call_once(self, _args: ()) -> Self::Output {
        let mut objs = vec![];
        loop {
            sleep(Duration::from_millis(1));
            // TODO: no magic
            match self.rx.recv_timeout(Duration::from_secs(10)) {
                Ok((recl, _)) => objs.push(recl),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    todo!()
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    todo!()
                }
            }

            let (gc, exited) = {
                let log = match self.access_log.read() {
                    Ok(log) => log,
                    Err(err) => todo!(),
                };
                // TODO: no magic
                let epochs: Vec<u64> = log.iter().map(|acc| acc.load(SeqCst)).collect();
                let gc = epochs.clone().into_iter().min().unwrap() - 10;
                let exited = epochs.into_iter().all(|acc| acc == 0);
                (gc, exited)
            };

            let mut new_objs = vec![];
            for obj in objs.into_iter() {
                match obj {
                    Reclaim::Value { epoch, .. } if epoch >= gc => new_objs.push(obj),
                    Reclaim::Entry { epoch, .. } if epoch >= gc => new_objs.push(obj),
                    _obj => (), // safe to drop the object here.
                }
            }

            objs = new_objs;

            if exited {
                todo!()
            }
        }
    }
}
