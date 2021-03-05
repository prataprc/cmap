use rand::{prelude::random, rngs::SmallRng, Rng, SeedableRng};
use structopt::StructOpt;

use std::{mem, thread, time};

use cmap::Map;

// TODO: when we compile bin/perf with `pprof` feature and run it via valgrind
//    there are memory-leaks. Is that normal ?

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

    #[cfg(feature = "pprof")]
    let guard = pprof::ProfilerGuard::new(100000).unwrap();

    let mut map: Map<Ky, u64> = Map::new(opts.threads + 1);
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
        let (opts, map) = (opts.clone(), map.cloned());
        let seed = seed + ((j as u128) * 100);
        let h = thread::spawn(move || do_incremental(j, seed, opts, map));
        handles.push(h);
    }

    for handle in handles.into_iter() {
        handle.join().unwrap()
    }

    #[cfg(feature = "pprof")]
    if let Ok(report) = guard.report().build() {
        let file = std::fs::File::create("flamegraph.svg").unwrap();
        report.flamegraph(file).unwrap();
    };

    println!("{:?}", map.validate());

    mem::drop(map)
}

fn do_incremental(j: usize, seed: u128, opts: Opt, mut map: Map<Ky, u64>) {
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
