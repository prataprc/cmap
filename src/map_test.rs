use arbitrary::{self, unstructured::Unstructured, Arbitrary};
use rand::{prelude::random, rngs::SmallRng, Rng, SeedableRng};

use std::{collections::BTreeMap, ops::Bound, thread};

use super::*;

type Ky = u8;

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
        match w % 2 {
            0 => assert_eq!(hamming_distance(w, bmp.clone()), Distance::Insert(o)),
            1 => assert_eq!(hamming_distance(w, bmp.clone()), Distance::Set(o)),
            _ => unreachable!(),
        }
    }

    let bmp = [
        0x55555555555555555555555555555555,
        0x55555555555555555555555555555555,
    ];
    for w in 0..=255 {
        let o = ((w % 128) / 2) as usize;
        match w % 2 {
            0 => assert_eq!(hamming_distance(w, bmp.clone()), Distance::Set(o)),
            1 => assert_eq!(hamming_distance(w, bmp.clone()), Distance::Insert(o + 1)),
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_map() {
    let seed: u128 = random();
    let seed: u128 = 108608880608704922882102056739567863183;
    println!("test_map seed {}", seed);
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let n_init = 1_000;
    let n_incr = 1_000;
    let n_threads = 8;

    let map: Map<Ky, u64> = Map::new();
    let mut btmap: BTreeMap<Ky, u64> = BTreeMap::new();

    let mut handles = vec![];
    let id = 0;
    let seed = seed + ((id as u128) * 100);
    let h = thread::spawn(move || with_btreemap(id, seed, n_init, map, btmap));
    handles.push(h);

    for handle in handles.into_iter() {
        btmap = handle.join().unwrap();
    }
}

fn with_btreemap(
    id: usize,
    seed: u128,
    n: usize,
    map: Map<Ky, u64>,
    mut btmap: BTreeMap<Ky, u64>,
) -> BTreeMap<Ky, u64> {
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let mut counts = [0_usize; 3];

    for _i in 0..n {
        let bytes = rng.gen::<[u8; 32]>();
        let mut uns = Unstructured::new(&bytes);

        let op: Op<Ky, u64> = uns.arbitrary().unwrap();
        println!("{}-op -- {:?}", id, op);
        match op.clone() {
            Op::Set(key, value) => {
                counts[0] += 1;
                assert_eq!(map.set(key, value).unwrap(), btmap.insert(key, value));
            }
            //Op::Remove(key) => {
            //    counts[1] += 1;
            //    assert_eq!(map.remove(&key), btmap.remove(&key));
            //}
            Op::Get(key) => {
                counts[2] += 1;
                println!("{:?} {:?}", map.get(&key), btmap.get(&key).cloned());
                assert_eq!(map.get(&key), btmap.get(&key).cloned());
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
    // Remove(K),
}
