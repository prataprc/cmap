use arbitrary::{self, unstructured::Unstructured, Arbitrary};
use rand::{prelude::random, rngs::SmallRng, Rng, SeedableRng};

use std::{cmp, mem, thread};

use super::*;

#[test]
fn test_arr_map() {
    let seed: u128 = random();
    // let seed: u128 = 268219686229904906077089108983355143992;
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let key_max = [1024 * 1024 * 1024, u32::MAX, 256, 16, 1024][rng.gen::<usize>() % 5];
    let n_ops = [1_000, 1_000_000, 10_000_000][rng.gen::<usize>() % 3];
    let n_threads = {
        let n = [1, 2, 4, 8, 16, 32, 64, 1024][rng.gen::<usize>() % 7];
        cmp::min(key_max, n)
    };
    let gc_period = [0, 1, 16, 32, 256, 1024][rng.gen::<usize>() % 6];
    let modul = key_max / n_threads;

    println!(
        "test_arr_map seed:{} key_max:{} ops:{} threads:{} modul:{}",
        seed, key_max, n_ops, n_threads, modul
    );

    let mut map: Map<u32, u32> = {
        let hash_builder = DefaultHasher::new();
        Map::new(n_threads as usize + 1, hash_builder)
    };
    map.set_gc_period(gc_period);
    map.print_sizing();

    let mut handles = vec![];
    for id in 0..n_threads {
        let seed = seed + ((id as u128) * 100);

        let map = map.cloned();
        let h = thread::spawn(move || with_arr(id, seed, modul, n_ops, map));

        handles.push(h);
    }

    let mut arr = vec![];
    for handle in handles.into_iter() {
        arr.extend_from_slice(&handle.join().unwrap());
    }

    for (key, val) in arr.iter().enumerate() {
        match val {
            0 => assert_eq!(map.get(&(key as u32)), None, "for key {}", key),
            _ => assert_eq!(map.get(&(key as u32)), Some(*val), "for key {}", key),
        }
    }

    println!("len {}", map.len());
    let mut len = 0;
    arr.iter().for_each(|v| len += if *v > 0 { 1 } else { 0 });
    assert_eq!(map.len(), len);
    println!("Validate .... {:?}", map.validate());

    // map.print();

    mem::drop(map);
    mem::drop(arr);
}

fn with_arr(
    id: u32,
    seed: u128,
    modul: u32,
    n_ops: usize,
    mut map: Map<u32, u32>,
) -> Vec<u32> {
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());
    let mut arr = vec![0_u32; modul as usize];

    let mut counts = [[0_usize; 2]; 3];

    for _i in 0..n_ops {
        let bytes = rng.gen::<[u8; 32]>();
        let mut uns = Unstructured::new(&bytes);

        let op: Op = uns.arbitrary().unwrap();
        let (off, op) = op.adjust_key(id, modul);
        // println!("{}-op -- {:?}", id, op);
        match op.clone() {
            Op::Set(key) => {
                // map.print();

                let arr_val = arr[off];
                arr[off] = arr[off].saturating_add(1);
                let map_val = map.set(key, arr[off]);
                if map_val.unwrap_or(0) != arr_val {
                    map.print();
                }

                counts[0][0] += 1;
                counts[0][1] += if map_val.is_none() { 0 } else { 1 };

                assert_eq!(map_val.unwrap_or(0), arr_val, "key {}", key);
            }
            Op::Remove(key) => {
                // map.print();

                let arr_val = arr[off];
                arr[off] = 0;
                let map_val = map.remove(&key);
                if map_val.unwrap_or(0) != arr_val {
                    map.print();
                }

                counts[1][0] += 1;
                counts[1][1] += if map_val.is_none() { 0 } else { 1 };

                assert_eq!(map_val.unwrap_or(0), arr_val, "key {}", key);
            }
            Op::Get(key) => {
                // map.print();

                let arr_val = arr[off];
                let map_val = map.get(&key);
                if map_val.unwrap_or(0) != arr_val {
                    map.print();
                }

                counts[2][0] += 1;
                counts[2][1] += if map_val.is_none() { 0 } else { 1 };

                assert_eq!(map_val.unwrap_or(0), arr_val, "key {}", key);
            }
        };
    }

    println!("{} counts {:?}", id, counts);
    arr
}

#[derive(Clone, Debug, Arbitrary)]
enum Op {
    Get(u32),
    Set(u32),
    Remove(u32),
}

impl Op {
    fn adjust_key(self, id: u32, modul: u32) -> (usize, Self) {
        match self {
            Op::Get(key) => {
                let off = key % modul;
                (off as usize, Op::Get((id * modul) + off))
            }
            Op::Set(key) => {
                let off = key % modul;
                (off as usize, Op::Set((id * modul) + off))
            }
            Op::Remove(key) => {
                let off = key % modul;
                (off as usize, Op::Remove((id * modul) + off))
            }
        }
    }
}
