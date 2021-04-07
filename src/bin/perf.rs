use dashmap::DashMap;
use flurry;
use rand::{prelude::random, rngs::SmallRng, Rng, SeedableRng};
use structopt::StructOpt;

use std::{mem, sync::Arc, thread, time};

use cmap::{Map, U32Hasher};

type Ky = u32;

/// Command line options.
#[derive(Clone, StructOpt)]
pub struct Opt {
    #[structopt(long = "seed")]
    seed: Option<u128>,

    #[structopt(long = "loads", default_value = "1000000")] // default 1M
    loads: usize,

    #[structopt(long = "gets", default_value = "0")] // default 1M
    gets: usize,

    #[structopt(long = "sets", default_value = "0")] // default 1M
    sets: usize,

    #[structopt(long = "rems", default_value = "0")] // default 1M
    rems: usize,

    #[structopt(long = "threads", default_value = "1")]
    threads: usize,

    #[structopt(long = "validate")]
    validate: bool,

    #[structopt(long = "dashmap")]
    dash_map: bool,

    #[structopt(long = "flurry")]
    flurry_map: bool,
}

fn main() {
    let opts = Opt::from_args();
    if opts.dash_map {
        dash_map(opts)
    } else if opts.flurry_map {
        flurry_map(opts)
    } else {
        cmap(opts)
    }
}

fn cmap(opts: Opt) {
    let seed = opts.seed.unwrap_or_else(random);
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let mut map: Map<Ky, u64, _> = Map::new(opts.threads + 1, U32Hasher::default());
    map.print_sizing();

    // initial load
    let start = time::Instant::now();
    for _i in 0..opts.loads {
        let key = rng.gen::<Ky>() % (opts.loads as Ky);
        let val: u64 = rng.gen();
        map.set(key, val);
    }

    println!("loaded {} items in {:?}", opts.loads, start.elapsed());

    let mut handles = vec![];
    for j in 0..opts.threads {
        let (opts, map) = (opts.clone(), map.clone());
        let seed = seed + ((j as u128) * 100);
        let h = thread::spawn(move || cmap_incremental(j, seed, opts, map));
        handles.push(h);
    }

    for handle in handles.into_iter() {
        handle.join().unwrap()
    }

    if opts.validate {
        println!("{:?}", map.validate());
    }

    mem::drop(map)
}

fn cmap_incremental(j: usize, seed: u128, opts: Opt, mut map: Map<Ky, u64, U32Hasher>) {
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let start = time::Instant::now();
    let (mut sets, mut rems, mut gets) = (opts.sets, opts.rems, opts.gets);
    let key_max = (opts.loads + opts.sets) as Ky;
    while (sets + rems + gets) > 0 {
        let key = rng.gen::<Ky>() % key_max;

        let op = rng.gen::<usize>() % (sets + rems + gets);
        if op < sets {
            map.set(key, rng.gen());
            sets -= 1;
        } else if op < (sets + rems) {
            map.remove(&key);
            rems -= 1;
        } else {
            map.get(&key);
            gets -= 1;
        }
    }
    println!(
        "incremental-{} for operations {}, took {:?}",
        j,
        opts.sets + opts.rems + opts.gets,
        start.elapsed()
    );
}

fn dash_map(opts: Opt) {
    let seed = opts.seed.unwrap_or_else(random);
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let dmap: Arc<DashMap<Ky, u64>> = Arc::new(DashMap::new());

    // initial load
    let start = time::Instant::now();
    for _i in 0..opts.loads {
        let key = rng.gen::<Ky>() % (opts.loads as Ky);
        let val: u64 = rng.gen();
        dmap.insert(key, val);
    }

    println!("loaded {} items in {:?}", opts.loads, start.elapsed());

    let mut handles = vec![];
    for j in 0..opts.threads {
        let (opts, dmap) = (opts.clone(), Arc::clone(&dmap));
        let seed = seed + ((j as u128) * 100);
        let h = thread::spawn(move || dmap_incremental(j, seed, opts, dmap));
        handles.push(h);
    }

    for handle in handles.into_iter() {
        handle.join().unwrap()
    }
}

fn dmap_incremental(j: usize, seed: u128, opts: Opt, dmap: Arc<DashMap<Ky, u64>>) {
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let start = time::Instant::now();
    let (mut sets, mut rems, mut gets) = (opts.sets, opts.rems, opts.gets);
    let key_max = (opts.loads + opts.sets) as Ky;
    while (sets + rems + gets) > 0 {
        let key = rng.gen::<Ky>() % key_max;

        let op = rng.gen::<usize>() % (sets + rems + gets);
        if op < sets {
            dmap.insert(key, rng.gen());
            sets -= 1;
        } else if op < (sets + rems) {
            dmap.remove(&key);
            rems -= 1;
        } else {
            dmap.get(&key);
            gets -= 1;
        }
    }
    println!(
        "incremental-{} for operations {}, took {:?}",
        j,
        opts.sets + opts.rems + opts.gets,
        start.elapsed()
    );
}

fn flurry_map(opts: Opt) {
    use flurry::HashMap;

    let seed = opts.seed.unwrap_or_else(random);
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let fmap: Arc<HashMap<Ky, u64>> = Arc::new(HashMap::new());

    // initial load
    let start = time::Instant::now();
    for _i in 0..opts.loads {
        let key = rng.gen::<Ky>() % (opts.loads as Ky);
        let val: u64 = rng.gen();
        fmap.pin().insert(key, val);
    }

    println!("loaded {} items in {:?}", opts.loads, start.elapsed());

    let mut handles = vec![];
    for j in 0..opts.threads {
        let (opts, fmap) = (opts.clone(), Arc::clone(&fmap));
        let seed = seed + ((j as u128) * 100);
        let h = thread::spawn(move || fmap_incremental(j, seed, opts, fmap));
        handles.push(h);
    }

    for handle in handles.into_iter() {
        handle.join().unwrap()
    }
}

fn fmap_incremental(
    j: usize,
    seed: u128,
    opts: Opt,
    fmap: Arc<flurry::HashMap<Ky, u64>>,
) {
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let start = time::Instant::now();
    let (mut sets, mut rems, mut gets) = (opts.sets, opts.rems, opts.gets);
    let key_max = (opts.loads + opts.sets) as Ky;
    while (sets + rems + gets) > 0 {
        let key = rng.gen::<Ky>() % key_max;

        let op = rng.gen::<usize>() % (sets + rems + gets);
        if op < sets {
            fmap.pin().insert(key, rng.gen());
            sets -= 1;
        } else if op < (sets + rems) {
            fmap.pin().remove(&key);
            rems -= 1;
        } else {
            fmap.pin().get(&key);
            gets -= 1;
        }
    }
    println!(
        "incremental-{} for operations {}, took {:?}",
        j,
        opts.sets + opts.rems + opts.gets,
        start.elapsed()
    );
}
