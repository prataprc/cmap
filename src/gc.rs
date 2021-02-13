use mkit::thread;

use std::{
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        mpsc, Arc, RwLock,
    },
    thread::sleep,
    time::Duration,
};

use crate::{entry::Entry, Error, Result};

// CAS operation

pub enum Mem<K, V> {
    Entry(*mut Entry<K, V>),
    // Node(*mut Node<K, V>),
    // Child(*mut Child<K, V>),
}

impl<K, V> Mem<K, V> {
    fn pass(self, epoch: u64, gc: &Gc<K, V>) {
        match self {
            //Mem::Node(ptr) => {
            //    let _node = unsafe { Box::from_raw(ptr) };
            //}
            //Mem::Child(ptr) => {
            //    let _child = unsafe { Box::from_raw(ptr) };
            //}
            Mem::Entry(ptr) => {
                let entry = unsafe { Box::from_raw(ptr) };
                let rclm = Reclaim::Entry { epoch, entry };
                err_at!(GcFail, gc.post(rclm)).unwrap();
            }
        }
    }

    fn fail(self) {
        match self {
            //Mem::Node(ptr) => {
            //    let _node = unsafe { Box::from_raw(ptr) };
            //}
            //Mem::Child(ptr) => {
            //    let _child = unsafe { Box::from_raw(ptr) };
            //}
            Mem::Entry(ptr) => {
                let _entry = unsafe { Box::from_raw(ptr) };
            }
        }
    }
}

pub struct Cas<'a, K, V> {
    epoch: u64,
    gc: &'a Gc<K, V>,
    pass: Vec<Mem<K, V>>,
    fail: Vec<Mem<K, V>>,
}

impl<'a, K, V> Cas<'a, K, V> {
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

    pub fn swing<T>(&mut self, loc: &AtomicPtr<T>, old: *mut T, new: *mut T) -> bool {
        if loc.compare_and_swap(old, new, SeqCst) == old {
            let epoch = self.epoch;
            let gc = &self.gc;
            self.pass.drain(..).for_each(|mem| mem.pass(epoch, gc));
            true
        } else {
            self.fail.drain(..).for_each(|mem| mem.fail());
            false
        }
    }
}

pub enum Reclaim<K, V> {
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
                    Reclaim::Entry { epoch, .. } if epoch >= gc => new_objs.push(obj),
                    Reclaim::Entry { .. } => (),
                }
            }

            objs = new_objs;

            if exited {
                todo!()
            }
        }
    }
}
