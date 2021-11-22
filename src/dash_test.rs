use arbitrary::{self, unstructured::Unstructured, Arbitrary};
use dashmap::DashMap;
use rand::{prelude::random, rngs::SmallRng, Rng, SeedableRng};

use std::{
    cmp, fmt, mem,
    ops::{Add, Mul, Rem},
    thread,
};

use super::*;

trait Key:
    Copy
    + Sized
    + Eq
    + Add<Output = Self>
    + Rem<Output = Self>
    + Mul<Output = Self>
    + fmt::Display
    + fmt::Debug
    + Hash
    + Arbitrary
{
}

impl Key for u8 {}
impl Key for u32 {}
impl Key for u64 {}
impl Key for u128 {}

macro_rules! test_code {
    ($seed:expr, $keytype:ty) => {{
        let mut rng = SmallRng::from_seed($seed.to_le_bytes());

        let n_ops = [1_000, 1_000_000, 10_000_000][rng.gen::<usize>() % 3];
        let key_max = [
            (((1024_u64 * 1024 * 1024) - 1) & ((<$keytype>::MAX - 1) as u64)) as $keytype,
            <$keytype>::MAX,
            255,
            16,
            ((1024_u64 - 1) & ((<$keytype>::MAX - 1) as u64)) as $keytype,
        ][rng.gen::<usize>() % 5];
        let n_threads = {
            let n = [1, 2, 4, 8, 16, 32, 64, 255][rng.gen::<usize>() % 7];
            cmp::min(key_max, n)
        };
        let gc_period = [0, 1, 16, 32, 256, 1024][rng.gen::<usize>() % 6];
        let modul = key_max / (n_threads as $keytype);

        println!(
            "test_dash_map_{} seed:{} key_max:{} ops:{} threads:{} modul:{}",
            stringify!($keytype),
            $seed,
            key_max,
            n_ops,
            n_threads,
            modul
        );

        let mut map: Map<$keytype, u64> = {
            let hash_builder = DefaultHasher::new();
            Map::new(n_threads as usize + 1, hash_builder)
        };
        map.set_gc_period(gc_period);
        map.print_sizing();
        let dmap: Arc<DashMap<$keytype, u64>> = Arc::new(DashMap::new());

        let mut handles = vec![];
        for id in 0..n_threads {
            let id = id as $keytype;
            let seed = $seed + ((id as u128) * 100);

            let (map, dmap) = (map.clone(), Arc::clone(&dmap));
            let h =
                thread::spawn(move || with_dashmap(id, seed, modul, n_ops, map, dmap));

            handles.push(h);
        }

        for handle in handles.into_iter() {
            handle.join().unwrap();
        }

        for item in dmap.iter() {
            let (key, val) = item.pair();
            assert_eq!(map.get(key), Some(*val), "for key {}", key);
        }

        println!("len {}", map.len());
        assert_eq!(map.len(), dmap.len());
        println!("Validate .... {:?}", map.validate());

        // map.print();

        mem::drop(map);
        mem::drop(dmap);
    }};
}

#[test]
fn test_with_dash_map_u8() {
    let seed: u128 =
        [221544245499661277858524746728600114414, random()][random::<usize>() % 2];
    // let seed: u128 = 221544245499661277858524746728600114414;

    test_code!(seed, u8);
}

#[test]
fn test_with_dash_map_u32() {
    let seed: u128 =
        [221544245499661277858524746728600114414, random()][random::<usize>() % 2];
    // let seed: u128 = 221544245499661277858524746728600114414;

    test_code!(seed, u32);
}

#[test]
fn test_with_dash_map_u64() {
    let seed: u128 =
        [221544245499661277858524746728600114414, random()][random::<usize>() % 2];
    // let seed: u128 = 221544245499661277858524746728600114414;

    test_code!(seed, u64);
}

#[test]
fn test_with_dash_map_u128() {
    let seed: u128 =
        [221544245499661277858524746728600114414, random()][random::<usize>() % 2];
    // let seed: u128 = 221544245499661277858524746728600114414;

    test_code!(seed, u128);
}

fn with_dashmap<K>(
    id: K,
    seed: u128,
    modul: K,
    n_ops: usize,
    mut map: Map<K, u64>,
    dmap: Arc<DashMap<K, u64>>,
) where
    K: Key,
{
    let mut rng = SmallRng::from_seed(seed.to_le_bytes());

    let mut counts = [[0_usize; 2]; 3];

    for _i in 0..n_ops {
        let bytes = rng.gen::<[u8; 32]>();
        let mut uns = Unstructured::new(&bytes);

        let mut op: Op<K> = uns.arbitrary().unwrap();
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
            Op::GetWith(key) => {
                // map.print();

                let map_val = map.get_with(&key, |v| *v);
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
enum Op<K>
where
    K: Key,
{
    Get(K),
    GetWith(K),
    Set(K, u64),
    Remove(K),
}

impl<K> Op<K>
where
    K: Key,
{
    fn adjust_key(self, id: K, modul: K) -> Self {
        match self {
            Op::Get(key) => {
                let key = (id * modul) + (key % modul);
                Op::Get(key)
            }
            Op::GetWith(key) => {
                let key = (id * modul) + (key % modul);
                Op::GetWith(key)
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
