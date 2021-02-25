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

pub struct Cas<'a, K, V> {
    epoch: &'a Arc<AtomicU64>,
    tx: &'a mpsc::Sender<Reclaim<K, V>>,
    pass: Vec<Mem<K, V>>,
    fail: Vec<Mem<K, V>>,
}

impl<'a, K, V> Cas<'a, K, V> {
    pub fn new(tx: &'a mpsc::Sender<Reclaim<K, V>>, epoch: &'a Arc<AtomicU64>) -> Self {
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
                tx.send(Reclaim::Child { epoch, child }).expect("ipc-fail");
            }
            Mem::Node(ptr) => {
                let node = unsafe { Box::from_raw(ptr) };
                tx.send(Reclaim::Node { epoch, node }).expect("ipc-fail");
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
            match obj {
                Reclaim::Child { epoch, .. } if epoch < gc => (),
                Reclaim::Node { epoch, .. } if epoch < gc => (),
                _ => new_objs.push(obj),
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

fn receive_blocks<K, V>(rx: &mpsc::Receiver<Reclaim<K, V>>) -> (Vec<Reclaim<K, V>>, bool) {
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
