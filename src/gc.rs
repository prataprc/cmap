use mkit::thread::{Rx, Tx};

use std::{
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        mpsc, Arc, RwLock,
    },
    thread::sleep,
    time::Duration,
};

use crate::{map::Child, map::Node};

// CAS operation

pub type Epochs = Arc<RwLock<Vec<Arc<AtomicU64>>>>;

pub struct Cas<'a, K, V> {
    epoch: Arc<AtomicU64>,
    at: Arc<AtomicU64>,
    gc: &'a Gc<K, V>,
    pass: Vec<Mem<K, V>>,
    fail: Vec<Mem<K, V>>,
}

impl<'a, K, V> Drop for Cas<'a, K, V> {
    fn drop(&mut self) {
        self.at.store(self.epoch.load(SeqCst), SeqCst);
    }
}

impl<'a, K, V> Cas<'a, K, V> {
    pub fn new(epoch: Arc<AtomicU64>, at: Arc<AtomicU64>, gc: &'a Gc<K, V>) -> Self {
        Cas {
            epoch,
            at,
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

    pub fn swing<T>(&mut self, loc: &AtomicPtr<T>, old: *mut T, new: *mut T) -> bool {
        if loc.compare_and_swap(old, new, SeqCst) == old {
            let epoch = self.epoch.load(SeqCst);
            let gc = &self.gc;
            self.pass.drain(..).for_each(|mem| mem.pass(epoch, gc));
            true
        } else {
            self.fail.drain(..).for_each(|mem| mem.fail());
            false
        }
    }
}

pub enum Mem<K, V> {
    // Entry(*mut Entry<K, V>), TODO
    Child(*mut Child<K, V>),
    Node(*mut Node<K, V>),
}

impl<K, V> Mem<K, V> {
    fn pass(self, epoch: u64, gc: &Gc<K, V>) {
        match self {
            // TODO
            //Mem::Entry(ptr) => {
            //    let entry = unsafe { Box::from_raw(ptr) };
            //    let rclm = Reclaim::Entry { epoch, entry };
            //    gc.post(rclm).expect("ipc-fail");
            //}
            Mem::Child(ptr) => {
                let child = unsafe { Box::from_raw(ptr) };
                let rclm = Reclaim::Child { epoch, child };
                gc.post(rclm).expect("ipc-fail");
            }
            Mem::Node(ptr) => {
                let node = unsafe { Box::from_raw(ptr) };
                let rclm = Reclaim::Node { epoch, node };
                gc.post(rclm).expect("ipc-fail");
            }
        }
    }

    fn fail(self) {
        match self {
            // TODO
            //Mem::Entry(ptr) => {
            //    let _entry = unsafe { Box::from_raw(ptr) };
            //}
            Mem::Child(ptr) => {
                let _child = unsafe { Box::from_raw(ptr) };
            }
            Mem::Node(ptr) => {
                let _node = unsafe { Box::from_raw(ptr) };
            }
        }
    }
}

pub enum Reclaim<K, V> {
    // Entry { epoch: u64, entry: Box<Entry<K, V>> }, TODO
    Child { epoch: u64, child: Box<Child<K, V>> },
    Node { epoch: u64, node: Box<Node<K, V>> },
}

pub type Gc<K, V> = Tx<Reclaim<K, V>>;

pub struct GcThread<K, V> {
    access_log: Epochs,
    rx: Rx<Reclaim<K, V>>,
}

impl<K, V> GcThread<K, V> {
    pub fn new(access_log: Epochs, rx: Rx<Reclaim<K, V>>) -> Self {
        GcThread { access_log, rx }
    }
}

impl<K, V> FnOnce<()> for GcThread<K, V> {
    type Output = ();

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
                let log = self.access_log.read().expect("fail-lock");
                // TODO: no magic
                let epochs: Vec<u64> = {
                    let iter = log.iter().map(|acc| acc.load(SeqCst));
                    iter.filter_map(|epoch| {
                        if epoch & 0x8000000000000000 == 0 {
                            None
                        } else {
                            Some(epoch & 0x7FFFFFFFFFFFFFFF)
                        }
                    })
                    .collect()
                };
                // TODO: no magic
                let gc = epochs.clone().into_iter().min().unwrap() - 10;
                let exited = epochs.into_iter().all(|acc| acc == 0);
                (gc, exited)
            };

            let mut new_objs = vec![];
            for obj in objs.into_iter() {
                match obj {
                    // TODO
                    //Reclaim::Entry { epoch, .. } if epoch >= gc => new_objs.push(obj),
                    //Reclaim::Entry { .. } => (),
                    Reclaim::Child { epoch, .. } if epoch >= gc => new_objs.push(obj),
                    Reclaim::Child { .. } => (),
                    Reclaim::Node { epoch, .. } if epoch >= gc => new_objs.push(obj),
                    Reclaim::Node { .. } => (),
                }
            }

            objs = new_objs;

            if exited {
                todo!()
            }
        }
    }
}
