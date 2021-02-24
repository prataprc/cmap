use arbitrary::{self, unstructured::Unstructured, Arbitrary};
use rand::{prelude::random, rngs::SmallRng, Rng, SeedableRng};

use std::{
    collections::BTreeMap,
    mem,
    ops::{Add, Div, Mul, Rem},
    thread,
};

use super::*;

type Ky = u16;

#[test]
fn test_list_operation() {
    let mut items = vec![
        Item {
            key: 20_u64,
            value: 200_u64,
        },
        Item {
            key: 10_u64,
            value: 100_u64,
        },
        Item {
            key: 50_u64,
            value: 500_u64,
        },
        Item {
            key: 30_u64,
            value: 300_u64,
        },
    ];

    assert_eq!(update_into_list(&10_u64, &1000_u64, &mut items), Some(100));
    assert_eq!(
        update_into_list(&10_u64, &10000_u64, &mut items),
        Some(1000)
    );
    assert_eq!(update_into_list(&60_u64, &600_u64, &mut items), None);

    let (items, item) = remove_from_list(&20_u64, &items).unwrap();
    assert_eq!(item, 200_u64);
    let (items, item) = remove_from_list(&60_u64, &items).unwrap();
    assert_eq!(item, 600_u64);
    assert_eq!(remove_from_list(&20_u64, &items), None);
    assert_eq!(remove_from_list(&60_u64, &items), None);

    assert_eq!(get_from_list(&10, &items), Some(10000_u64));
    assert_eq!(get_from_list(&50, &items), Some(500_u64));
    assert_eq!(get_from_list(&30, &items), Some(300_u64));
    assert_eq!(get_from_list(&20, &items), None);

    assert_eq!(
        items,
        vec![
            Item {
                key: 10_u64,
                value: 10000_u64,
            },
            Item {
                key: 50_u64,
                value: 500_u64,
            },
            Item {
                key: 30_u64,
                value: 300_u64,
            },
        ]
    );
}

#[test]
fn test_hamming_distance() {
    let bmp = [
        0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
    ];
    for w in 0..=255 {
        let o = ((w % 128) / 2) as usize;
        let dist = hamming_distance(w, bmp.clone());
        match w % 2 {
            0 if w < 128 => assert_eq!(dist, Distance::Insert(o)),
            0 => assert_eq!(dist, Distance::Insert(64 + o)),
            1 if w < 128 => assert_eq!(dist, Distance::Set(o)),
            1 => assert_eq!(dist, Distance::Set(64 + o)),
            _ => unreachable!(),
        }
    }

    let bmp = [
        0x55555555555555555555555555555555,
        0x55555555555555555555555555555555,
    ];
    for w in 0..=255 {
        let o = ((w % 128) / 2) as usize;
        let dist = hamming_distance(w, bmp.clone());
        match w % 2 {
            0 if w < 128 => assert_eq!(dist, Distance::Set(o)),
            0 => assert_eq!(dist, Distance::Set(64 + o)),
            1 if w < 128 => assert_eq!(dist, Distance::Insert(o + 1)),
            1 => assert_eq!(dist, Distance::Insert(64 + o + 1)),
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_map() {
    let seed: u128 = random();
    let seed: u128 = 108608880608704922882102056739567863183;
    println!("test_map seed {}", seed);

    let n_ops = 2_000_000; // TODO
    let n_threads = 8; // TODO
    let modul = Ky::MAX / n_threads;

    let map: Map<Ky, u64> = Map::new();
    let mut handles = vec![];
    for id in 0..n_threads {
        let seed = seed + ((id as u128) * 100);

        let map = map.cloned();
        let btmap: BTreeMap<Ky, u64> = BTreeMap::new();
        let h = thread::spawn(move || with_btreemap(id, seed, modul, n_ops, map, btmap));

        handles.push(h);
    }

    let mut btmaps = vec![];
    for handle in handles.into_iter() {
        btmaps.push(handle.join().unwrap());
    }

    //assert_eq!(map.len(), btmaps[0].len());

    //for (key, val) in btmaps[0].iter() {
    //    assert_eq!(map.get(key), Some(val.clone()));
    //}
    mem::drop(map);
    mem::drop(btmaps);
    loop {}
}

fn with_btreemap(
    id: Ky,
    seed: u128,
    modul: Ky,
    n_ops: usize,
    map: Map<Ky, u64>,
    mut btmap: BTreeMap<Ky, u64>,
) -> BTreeMap<Ky, u64> {
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let mut counts = [0_usize; 3];

    for _i in 0..n_ops {
        let bytes = rng.gen::<[u8; 32]>();
        let mut uns = Unstructured::new(&bytes);

        let mut op: Op<Ky, u64> = uns.arbitrary().unwrap();
        op = op.adjust_key(id, modul, 1);
        // println!("{}-op -- {:?}", id, op);
        match op.clone() {
            Op::Set(key, value) => {
                // map.print();

                counts[0] += 1;

                let map_val = map.set(key, value).unwrap();
                let btmap_val = btmap.insert(key, value);
                if map_val != btmap_val {
                    map.print();
                }
                assert_eq!(map_val, btmap_val, "key {}", key);
            }
            Op::Remove(key) => {
                // map.print();

                counts[1] += 1;

                let map_val = map.remove(&key);
                let btmap_val = btmap.remove(&key);
                if map_val != btmap_val {
                    map.print();
                }
                assert_eq!(map_val, btmap_val, "key {}", key);
            }
            Op::Get(key) => {
                // map.print();

                counts[2] += 1;

                let map_val = map.get(&key);
                let btmap_val = btmap.get(&key).cloned();
                // println!("{:?} {:?}", map_val, btmap_val);
                if map_val != btmap_val {
                    map.print();
                }
                assert_eq!(map_val, btmap_val, "key {}", key);
            }
        };
    }

    println!("{} counts {:?}", id, counts);
    btmap
}

#[derive(Clone, Debug, Arbitrary)]
enum Op<K, V> {
    Get(K),
    Set(K, V),
    Remove(K),
}

impl<K, V> Op<K, V>
where
    K: Copy + Mul<Output = K> + Rem<Output = K> + Add<Output = K> + Div<Output = K>,
{
    fn adjust_key(self, id: K, modul: K, div: K) -> Self {
        match self {
            Op::Get(key) => {
                let key = key / div;
                let key = (id * modul) + (key % modul);
                Op::Get((id * modul) + (key % modul))
            }
            Op::Set(key, value) => {
                let key = key / div;
                let key = (id * modul) + (key % modul);
                Op::Set(key, value)
            }
            Op::Remove(key) => {
                let key = key / div;
                let key = (id * modul) + (key % modul);
                Op::Remove(key)
            }
        }
    }
}
