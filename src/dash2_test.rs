use arbitrary::{self, unstructured::Unstructured, Arbitrary};
use dashmap::DashMap;
use rand::{prelude::random, rngs::StdRng, Rng, SeedableRng};

use std::ops::{Add, Mul, Rem};
use std::{cmp, fmt, mem, thread};

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
        let mut rng = StdRng::seed_from_u64($seed);

        let n_ops = [1_000, 1_000_000][rng.gen::<usize>() % 2];
        let key_max = [
            (((1024_u64 * 1024 * 1024) - 1) & ((<$keytype>::MAX - 1) as u64)) as $keytype,
            <$keytype>::MAX,
            255,
            16,
            ((1024_u64 - 1) & ((<$keytype>::MAX - 1) as u64)) as $keytype,
        ][rng.gen::<usize>() % 5];
        let n_threads = {
            let n = [1, 2, 4, 8, 16][rng.gen::<usize>() % 5];
            cmp::min(key_max, n)
        };
        let gc_period = [0, 1, 16, 32, 256, 1024][rng.gen::<usize>() % 6];
        let modul = key_max / (n_threads as $keytype);

        println!(
            "test_dash2_map_{} seed:{} key_max:{} ops:{} threads:{} modul:{}",
            stringify!($keytype),
            $seed,
            key_max,
            n_ops,
            n_threads,
            modul
        );

        let mut map: Map<$keytype, u128> = {
            let hash_builder = DefaultHasher::new();
            Map::new(n_threads as usize + 1, hash_builder)
        };
        map.set_gc_period(gc_period);
        map.print_sizing();
        let dmap: Arc<DashMap<$keytype, u128>> = Arc::new(DashMap::new());

        let mut handles = vec![];
        for id in 0..n_threads {
            let id = id as $keytype;
            let seed = $seed + ((id as u64) * 100);

            let (map, dmap) = (map.cloned(), Arc::clone(&dmap));
            let h = thread::spawn(move || with_dashmap(id, seed, n_ops, map, dmap));

            handles.push(h);
        }

        for handle in handles.into_iter() {
            handle.join().unwrap();
        }

        println!(
            "test_dash2_map_{} Validate .... {:?}",
            stringify!($keytype),
            map.validate()
        );

        let mut mismatch_count = 0;
        for item in dmap.iter() {
            let (key, val) = item.pair();
            if map.get(key) != Some(*val) {
                mismatch_count += 1;
            }
        }

        println!(
            "test_dash2_map_{} len map:{} dash:{}",
            stringify!($keytype),
            map.len(),
            dmap.len()
        );

        println!(
            "test_dash2_map_{} mismatch_count:{}",
            stringify!($keytype),
            mismatch_count
        );

        mismatch_count += if map.len() != dmap.len() { 1 } else { 0 };
        match stringify!($keytype) {
            "u8" if mismatch_count > 2 => panic!("mismatch_count:{} < 2", mismatch_count),
            "u8" => (),
            keytype => assert!(
                mismatch_count == 0,
                "mismatch_count:{} type:{}",
                mismatch_count,
                keytype
            ),
        }

        // map.print();

        mem::drop(map);
        mem::drop(dmap);
    }};
}

#[test]
fn test_with_dash_map_u8() {
    let seed: u64 = random();

    test_code!(seed, u8);
}

#[test]
fn test_with_dash_map_u32() {
    let seed: u64 = random();

    test_code!(seed, u32);
}

#[test]
fn test_with_dash_map_u64() {
    let seed: u64 = random();

    test_code!(seed, u64);
}

#[test]
fn test_with_dash_map_u128() {
    let seed: u64 = random();

    test_code!(seed, u128);
}

fn with_dashmap<K>(
    id: K,
    seed: u64,
    n_ops: usize,
    mut map: Map<K, u128>,
    dmap: Arc<DashMap<K, u128>>,
) where
    K: Key,
{
    let mut rng = StdRng::seed_from_u64(seed);

    let mut counts = [[0_usize; 2]; 3];

    for _i in 0..n_ops {
        let bytes = rng.gen::<[u8; 32]>();
        let mut uns = Unstructured::new(&bytes);

        let mut op: Op<K> = uns.arbitrary().unwrap();
        op = op.adjust_op();
        // println!("{}-op -- {:?}", id, op);
        match op.clone() {
            Op::Set(key, value) => {
                // map.print();

                let map_val = map.set(key, value).map(|x| x as i128);
                let _dmap_val = dmap.insert(key, value).map(|x| x as i128);

                counts[0][0] += 1;
                counts[0][1] += if map_val.is_none() { 0 } else { 1 };

                // assert_eq!(map_val, dmap_val, "key {}", key);
            }
            Op::Remove(key) => {
                // map.print();

                let map_val = map.remove(&key);
                let _dmap_val = dmap.remove(&key).map(|(_, v)| v);

                counts[1][0] += 1;
                counts[1][1] += if map_val.is_none() { 0 } else { 1 };

                // assert_eq!(map_val, dmap_val, "key {}", key);
            }
            Op::Get(key) => {
                // map.print();

                let map_val = map.get(&key).map(|x| x as i128);
                let _dmap_val = dmap.get(&key).map(|x| *x as i128);

                counts[2][0] += 1;
                counts[2][1] += if map_val.is_none() { 0 } else { 1 };

                // assert_eq!(map_val, dmap_val, "key {}", key);
            }
            Op::GetWith(key) => {
                // map.print();

                let map_val = map.get_with(&key, |v| *v).map(|x| x as i128);
                let _dmap_val = dmap.get(&key).map(|x| *x as i128);

                counts[2][0] += 1;
                counts[2][1] += if map_val.is_none() { 0 } else { 1 };

                // assert_eq!(map_val, dmap_val, "key {}", key);
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
    Set(K, u128),
    Remove(K),
}

impl<K> Op<K>
where
    K: Key,
{
    // we will allow same keys to be used by other threads, but use Instant::now()
    // to track concurrency issues.
    fn adjust_op(self) -> Self {
        match self {
            Op::Get(key) => Op::Get(key),
            Op::GetWith(key) => Op::GetWith(key),
            Op::Set(key, _) => {
                let value = time::SystemTime::now()
                    .duration_since(time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                Op::Set(key, value)
            }
            Op::Remove(key) => Op::Remove(key),
        }
    }
}
