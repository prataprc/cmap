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

pub struct Epoch {
    epoch: Arc<AtomicU64>,
    at: Arc<AtomicU64>,
}

impl Epoch {
    pub fn new(epoch: Arc<AtomicU64>, at: Arc<AtomicU64>) -> Epoch {
        at.store(epoch.load(SeqCst) | 0x8000000000000000, SeqCst);
        Epoch { epoch, at }
    }
}

impl Drop for Epoch {
    fn drop(&mut self) {
        self.at.store(self.epoch.load(SeqCst), SeqCst);
    }
}

pub struct Cas<'a, K, V> {
    epoch: Arc<AtomicU64>,
    tx: &'a mpsc::Sender<Reclaim<K, V>>,
    pass: Vec<Mem<K, V>>,
    fail: Vec<Mem<K, V>>,
}

impl<'a, K, V> Cas<'a, K, V> {
    pub fn new(tx: &'a mpsc::Sender<Reclaim<K, V>>, epoch: Arc<AtomicU64>) -> Self {
        Cas {
            epoch,
            tx,
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
            let tx = &self.tx;
            self.pass.drain(..).for_each(|mem| mem.pass(epoch, tx));
            true
        } else {
            self.fail.drain(..).for_each(|mem| mem.fail());
            false
        }
    }
}

pub enum Mem<K, V> {
    Child(*mut Child<K, V>),
    Node(*mut Node<K, V>),
}

impl<K, V> Mem<K, V> {
    fn pass(self, epoch: u64, tx: &mpsc::Sender<Reclaim<K, V>>) {
        match self {
            Mem::Child(ptr) => {
                let child = unsafe { Box::from_raw(ptr) };
                let rclm = Reclaim::Child { epoch, child };
                tx.send(rclm).expect("ipc-fail");
            }
            Mem::Node(ptr) => {
                let node = unsafe { Box::from_raw(ptr) };
                let rclm = Reclaim::Node { epoch, node };
                tx.send(rclm).expect("ipc-fail");
            }
        }
    }

    fn fail(self) {
        match self {
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
    Child { epoch: u64, child: Box<Child<K, V>> },
    Node { epoch: u64, node: Box<Node<K, V>> },
}

pub fn gc_thread<K, V>(
    epoch: Arc<AtomicU64>,
    access_log: Arc<RwLock<Vec<Arc<AtomicU64>>>>,
    rx: mpsc::Receiver<Reclaim<K, V>>,
) {
    let mut objs = vec![];
    loop {
        sleep(Duration::from_millis(1));
        // TODO: no magic number
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(recl) => objs.push(recl),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                todo!()
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                todo!()
            }
        }

        let (gc, exited) = {
            let log = access_log.read().expect("fail-lock");
            // TODO: no magic
            let epochs: Vec<u64> = {
                let iter = log.iter().map(|acc| acc.load(SeqCst));
                iter.filter_map(|el| {
                    if el & 0x8000000000000000 == 0 {
                        Some(el & 0x7FFFFFFFFFFFFFFF)
                    } else {
                        Some(epoch.load(SeqCst))
                    }
                })
                .collect()
            };
            // TODO: no magic
            let gc = match epochs.clone().into_iter().min() {
                Some(gc) => gc.saturating_sub(10),
                None => continue,
            };
            let exited = epochs.into_iter().all(|acc| acc == 0);
            (gc, exited)
        };

        let mut new_objs = vec![];
        for obj in objs.into_iter() {
            match obj {
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
