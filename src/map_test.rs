use arbitrary::{self, unstructured::Unstructured, Arbitrary};
use rand::{prelude::random, rngs::SmallRng, Rng, SeedableRng};

use std::{cmp, collections::BTreeMap, mem, thread};

use super::*;

type Ky = u32;

#[test]
fn test_list_operation() {
    let mut items: Vec<Item<Ky, u64>> = vec![
        Item {
            key: 20,
            value: 200,
        },
        Item {
            key: 10,
            value: 100,
        },
        Item {
            key: 50,
            value: 500,
        },
        Item {
            key: 30,
            value: 300,
        },
    ];

    assert_eq!(update_into_list((10, 1000).into(), &mut items), Some(100));
    assert_eq!(update_into_list((10, 10000).into(), &mut items), Some(1000));
    assert_eq!(update_into_list((60, 600).into(), &mut items), None);

    assert_eq!(get_from_list(&10, &items), Some(10000));
    assert_eq!(get_from_list(&50, &items), Some(500));
    assert_eq!(get_from_list(&30, &items), Some(300));
    assert_eq!(get_from_list(&20, &items), Some(200));

    assert_eq!(
        items,
        vec![
            Item {
                key: 20,
                value: 200,
            },
            Item {
                key: 10,
                value: 10000,
            },
            Item {
                key: 50,
                value: 500,
            },
            Item {
                key: 30,
                value: 300,
            },
            Item {
                key: 60,
                value: 600,
            },
        ]
    );
}

#[test]
fn test_hamming_distance() {
    let bmp = 0xaaaa;
    for w in 0..16 {
        let dist = hamming_distance(w, bmp.clone());
        let o = ((w % 16) / 2) as usize;
        match w % 2 {
            0 => assert_eq!(dist, Distance::Insert(o)),
            1 => assert_eq!(dist, Distance::Set(o)),
            _ => unreachable!(),
        }
    }

    let bmp = 0x5555;
    for w in 0..16 {
        let o = ((w % 16) / 2) as usize;
        let dist = hamming_distance(w, bmp.clone());
        match w % 2 {
            0 => assert_eq!(dist, Distance::Set(o)),
            1 => assert_eq!(dist, Distance::Insert(o + 1)),
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_map() {
    let seed: u128 =
        [93808280188270876915817886943766741423, random()][random::<usize>() % 2];
    // let seed: u128 = 93808280188270876915817886943766741423;
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

    let mut handles = vec![];
    for id in 0..n_threads {
        let seed = seed + ((id as u128) * 100);

        let map = map.cloned();
        let btmap: BTreeMap<Ky, u64> = BTreeMap::new();
        let h = thread::spawn(move || with_btreemap(id, seed, modul, n_ops, map, btmap));

        handles.push(h);
    }

    let mut btmap: BTreeMap<Ky, u64> = BTreeMap::new();
    for handle in handles.into_iter() {
        btmap = merge_btmap([btmap, handle.join().unwrap()]);
    }

    for (key, val) in btmap.iter() {
        assert_eq!(map.get(key), Some(val.clone()), "for key {}", key);
    }

    let ln = map.len();
    println!("len {}", ln);
    assert_eq!(ln, btmap.len());
    println!("Validate .... {:?}", map.validate());

    // map.print();

    mem::drop(map);
    mem::drop(btmap);
}

fn with_btreemap(
    id: Ky,
    seed: u128,
    modul: Ky,
    n_ops: usize,
    mut map: Map<Ky, u64>,
    mut btmap: BTreeMap<Ky, u64>,
) -> BTreeMap<Ky, u64> {
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
                let btmap_val = btmap.insert(key, value);
                if map_val != btmap_val {
                    // map.print();
                }

                counts[0][0] += 1;
                counts[0][1] += if map_val.is_none() { 0 } else { 1 };

                assert_eq!(map_val, btmap_val, "key {}", key);
            }
            Op::Remove(key) => {
                // map.print();

                let map_val = map.remove(&key);
                let btmap_val = btmap.remove(&key);
                if map_val != btmap_val {
                    // map.print();
                }

                counts[1][0] += 1;
                counts[1][1] += if map_val.is_none() { 0 } else { 1 };

                assert_eq!(map_val, btmap_val, "key {}", key);
            }
            Op::Get(key) => {
                // map.print();

                let map_val = map.get(&key);
                let btmap_val = btmap.get(&key).cloned();
                if map_val != btmap_val {
                    // map.print();
                }

                counts[2][0] += 1;
                counts[2][1] += if map_val.is_none() { 0 } else { 1 };

                assert_eq!(map_val, btmap_val, "key {}", key);
            }
        };
    }

    println!("{} counts {:?}", id, counts);
    btmap
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

fn merge_btmap(items: [BTreeMap<Ky, u64>; 2]) -> BTreeMap<Ky, u64> {
    let [mut one, two] = items;

    for (key, value) in two.iter() {
        one.insert(*key, *value);
    }
    one
}
