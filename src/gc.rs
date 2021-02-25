use std::{
    mem,
    sync::{
        atomic::{AtomicPtr, AtomicU64, Ordering::SeqCst},
        mpsc, Arc, RwLock,
    },
    thread, time,
};

use crate::{map::Child, map::Node};

const EPOCH_PERIOD: time::Duration = time::Duration::from_millis(10);
const ENTER_MASK: u64 = 0x8000000000000000;
const EPOCH_MASK: u64 = 0x7FFFFFFFFFFFFFFF;

// CAS operation

pub struct Epoch {
    epoch: Arc<AtomicU64>,
    at: Arc<AtomicU64>,
}

impl Epoch {
    pub fn new(epoch: Arc<AtomicU64>, at: Arc<AtomicU64>) -> Epoch {
        at.store(epoch.load(SeqCst) | ENTER_MASK, SeqCst);
        Epoch { epoch, at }
    }
}

impl Drop for Epoch {
    fn drop(&mut self) {
        self.at.store(self.epoch.load(SeqCst), SeqCst);
    }
}

pub struct Cas<'a, V> {
    epoch: &'a Arc<AtomicU64>,
    tx: &'a mpsc::Sender<Reclaim<V>>,
    pass: Vec<Mem<V>>,
    fail: Vec<Mem<V>>,
}

impl<'a, V> Cas<'a, V> {
    pub fn new(tx: &'a mpsc::Sender<Reclaim<V>>, epoch: &'a Arc<AtomicU64>) -> Self {
        Cas {
            epoch,
            tx,
            pass: Vec::default(),
            fail: Vec::default(),
        }
    }

    pub fn free_on_pass(&mut self, m: Mem<V>) {
        self.pass.push(m)
    }

    pub fn free_on_fail(&mut self, m: Mem<V>) {
        self.fail.push(m)
    }

    pub fn swing<T>(self, loc: &AtomicPtr<T>, old: *mut T, new: *mut T) -> bool {
        if loc.compare_and_swap(old, new, SeqCst) == old {
            let epoch = Some(self.epoch.load(SeqCst));
            let items = OwnedMem::new_vec(self.pass);
            self.tx.send(Reclaim { epoch, items }).ok(); // TODO: handle result
            true
        } else {
            for item in self.fail.into_iter() {
                match item {
                    Mem::Child(ptr) => {
                        let _child = unsafe { Box::from_raw(ptr) };
                    }
                    Mem::Node(ptr) => {
                        let _node = unsafe { Box::from_raw(ptr) };
                    }
                }
            }
            false
        }
    }
}

pub enum Mem<V> {
    Child(*mut Child<V>),
    Node(*mut Node<V>),
}

pub struct Reclaim<V> {
    epoch: Option<u64>,
    items: Vec<OwnedMem<V>>,
}

enum OwnedMem<V> {
    Child(Box<Child<V>>),
    Node(Box<Node<V>>),
}

impl<V> OwnedMem<V> {
    fn new_vec(mems: Vec<Mem<V>>) -> Vec<Self> {
        mems.into_iter()
            .map(|m| match m {
                Mem::Child(ptr) => unsafe { OwnedMem::Child(Box::from_raw(ptr)) },
                Mem::Node(ptr) => unsafe { OwnedMem::Node(Box::from_raw(ptr)) },
            })
            .collect()
    }
}

pub fn gc_thread<V>(
    epoch: Arc<AtomicU64>,
    access_log: Arc<RwLock<Vec<Arc<AtomicU64>>>>,
    rx: mpsc::Receiver<Reclaim<V>>,
) {
    let mut objs = vec![];

    loop {
        thread::sleep(EPOCH_PERIOD);
        let (mut fresh, exit) = receive_blocks(&rx);
        objs.append(&mut fresh);
        if exit {
            break;
        }

        let (gc, exited) = {
            let log = access_log.read().expect("fail-lock");
            let (epochs, exited): (Vec<u64>, bool) = {
                let epochs: Vec<u64> = log.iter().map(|acc| acc.load(SeqCst)).collect();
                (
                    epochs
                        .clone()
                        .into_iter()
                        .map(|el| {
                            if el & ENTER_MASK == 0 {
                                epoch.load(SeqCst)
                            } else {
                                el & EPOCH_MASK
                            }
                        })
                        .collect(),
                    epochs.into_iter().all(|epoch| epoch == 0),
                )
            };
            let gc = match epochs.clone().into_iter().min() {
                Some(gc) => gc,
                None => continue,
            };
            (gc, exited)
        };

        if exited {
            break;
        }

        let _n = objs.len();

        let mut new_objs = vec![];
        for obj in objs.into_iter() {
            match obj.epoch {
                Some(epoch) if epoch < gc => mem::drop(obj.items),
                Some(_) | None => new_objs.push(obj),
            }
        }
        objs = new_objs;

        //#[cfg(test)] // TODO make it debug feature
        //println!("garbage collected epoch:{} {}/{}", gc, _n, objs.len());
    }

    #[cfg(test)]
    println!("exiting with pending allocs {}", objs.len());
    mem::drop(objs);
}

fn receive_blocks<V>(rx: &mpsc::Receiver<Reclaim<V>>) -> (Vec<Reclaim<V>>, bool) {
    let mut objs = vec![];
    loop {
        match rx.try_recv() {
            Ok(recl) => objs.push(recl),
            Err(mpsc::TryRecvError::Empty) => return (objs, false),
            Err(mpsc::TryRecvError::Disconnected) => {
                #[cfg(test)]
                println!("exiting epoch-gc, disconnected");
                return (objs, true);
            }
        }
    }
}
