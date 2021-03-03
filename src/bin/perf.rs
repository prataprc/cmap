use rand::{prelude::random, rngs::SmallRng, Rng, SeedableRng};
use structopt::StructOpt;

use std::{thread, time};

use cmap::Map;

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
}

fn main() {
    let opts = Opt::from_args();
    let seed = opts.seed.unwrap_or_else(random);
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let mut map: Map<Ky, u64> = Map::new();

    // initial load
    let start = time::Instant::now();
    for _i in 0..opts.loads {
        let (key, val): (Ky, u64) = (rng.gen(), rng.gen());
        map.set(key % 1_000_000, val);
    }

    println!("loaded {} items in {:?}", opts.loads, start.elapsed());

    let mut handles = vec![];
    for j in 0..opts.threads {
        let (opts, map) = (opts.clone(), map.cloned());
        let seed = seed + ((j as u128) * 100);
        let h = thread::spawn(move || do_incremental(j, seed, opts, map));
        handles.push(h);
    }

    for handle in handles.into_iter() {
        handle.join().unwrap()
    }

    println!("{:?}", map.collisions());

    println!("{:?}", map.validate());
}

fn do_incremental(j: usize, seed: u128, opts: Opt, mut map: Map<Ky, u64>) {
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let start = time::Instant::now();
    let (mut sets, mut rems, mut gets) = (opts.sets, opts.rems, opts.gets);
    while (sets + rems + gets) > 0 {
        let key = rng.gen::<Ky>() % 1_000_000;

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
