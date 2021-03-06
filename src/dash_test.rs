use arbitrary::{self, unstructured::Unstructured, Arbitrary};
use dashmap::DashMap;
use rand::{prelude::random, rngs::SmallRng, Rng, SeedableRng};

use std::{cmp, mem, thread};

use super::*;

type Ky = u32;

#[test]
fn test_dash_map() {
    let seed: u128 = random();
    // let seed: u128 = 114474774555146480506885522408182975209;
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let key_max = [1024 * 1024 * 1024, Ky::MAX, 256, 16, 1024][rng.gen::<usize>() % 5];
    let n_ops = [1_000, 1_000_000, 10_000_000][rng.gen::<usize>() % 3];
    let n_threads = {
        let n = [1, 2, 4, 8, 16, 32, 64, 1024][rng.gen::<usize>() % 7];
        cmp::min(key_max, n)
    };
    let modul = key_max / n_threads;

    println!(
        "test_map seed:{} key_max:{} ops:{} threads:{} modul:{}",
        seed, key_max, n_ops, n_threads, modul
    );

    let mut map: Map<Ky, u64> = Map::new(n_threads as usize + 1);
    map.print_sizing();
    let dmap: Arc<DashMap<Ky, u64>> = Arc::new(DashMap::new());

    let mut handles = vec![];
    for id in 0..n_threads {
        let seed = seed + ((id as u128) * 100);

        let (map, dmap) = (map.cloned(), Arc::clone(&dmap));
        let h = thread::spawn(move || with_dashmap(id, seed, modul, n_ops, map, dmap));

        handles.push(h);
    }

    for handle in handles.into_iter() {
        handle.join().unwrap();
    }

    for item in dmap.iter() {
        let (key, val) = item.pair();
        assert_eq!(map.get(key), Some(val.clone()), "for key {}", key);
    }

    println!("len {}", map.len());
    assert_eq!(map.len(), dmap.len());
    println!("Validate .... {:?}", map.validate());

    // map.print();

    mem::drop(map);
    mem::drop(dmap);
}

fn with_dashmap(
    id: Ky,
    seed: u128,
    modul: Ky,
    n_ops: usize,
    mut map: Map<Ky, u64>,
    dmap: Arc<DashMap<Ky, u64>>,
) {
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let mut counts = [[0_usize; 2]; 3];

    for _i in 0..n_ops {
        let bytes = rng.gen::<[u8; 32]>();
        let mut uns = Unstructured::new(&bytes);

        let mut op: Op = uns.arbitrary().unwrap();
        op = op.adjust_key(id, modul);
        // println!("{}-op -- {:?}", id, op);
        match op.clone() {
            Op::Set(key, value) => {
                // map.print();

                let map_val = map.set(key, value);
                let dmap_val = dmap.insert(key, value);
                if map_val != dmap_val {
                    map.print();
                }

                counts[0][0] += 1;
                counts[0][1] += if map_val.is_none() { 0 } else { 1 };

                assert_eq!(map_val, dmap_val, "key {}", key);
            }
            Op::Remove(key) => {
                // map.print();

                let map_val = map.remove(&key);
                let dmap_val = dmap.remove(&key).map(|(_, v)| v);
                if map_val != dmap_val {
                    map.print();
                }

                counts[1][0] += 1;
                counts[1][1] += if map_val.is_none() { 0 } else { 1 };

                assert_eq!(map_val, dmap_val, "key {}", key);
            }
            Op::Get(key) => {
                // map.print();

                let map_val = map.get(&key);
                let dmap_val = dmap.get(&key).map(|x| *x);
                if map_val != dmap_val {
                    map.print();
                }

                counts[2][0] += 1;
                counts[2][1] += if map_val.is_none() { 0 } else { 1 };

                assert_eq!(map_val, dmap_val, "key {}", key);
            }
        };
    }

    println!("{} counts {:?}", id, counts);
}

#[derive(Clone, Debug, Arbitrary)]
enum Op {
    Get(Ky),
    Set(Ky, u64),
    Remove(Ky),
}

impl Op {
    fn adjust_key(self, id: Ky, modul: Ky) -> Self {
        match self {
            Op::Get(key) => {
                let key = (id * modul) + (key % modul);
                Op::Get(key)
            }
            Op::Set(key, value) => {
                let key = (id * modul) + (key % modul);
                Op::Set(key, value)
            }
            Op::Remove(key) => {
                let key = (id * modul) + (key % modul);
                Op::Remove(key)
            }
        }
    }
}
