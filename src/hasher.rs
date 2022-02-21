// Module provies hasher implementation for generating key digest.
//
// * **[DefaultHasher]**, wraps google's city hash for using it with [crate::Map]
// * **[U32Hasher]**, can be used for [u32] type keys, where keys are directly
//   returned as the hash digest.

use std::{
    convert::TryInto,
    hash::{BuildHasher, Hasher},
};

/// Type uses google's city hash to convert [Hash]able key into ``u32``.
/// Refer [cityhash_rs] for details.
#[derive(Clone, Copy, Default)]
pub struct DefaultHasher {
    city_hash: u128,
}

impl DefaultHasher {
    pub fn new() -> DefaultHasher {
        DefaultHasher::default()
    }
}

impl BuildHasher for DefaultHasher {
    type Hasher = Self;

    #[inline]
    fn build_hasher(&self) -> Self {
        *self
    }
}

impl Hasher for DefaultHasher {
    fn finish(&self) -> u64 {
        ((self.city_hash >> 64) as u64) ^ ((self.city_hash & 0xFFFFFFFFFFFFFFFF) as u64)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.city_hash = cityhash_rs::cityhash_110_128(bytes);
    }
}

/// Type implement [BuildHasher] optimized for ``u32`` key set.
#[derive(Clone, Default)]
pub struct U32Hasher {
    key: u32,
}

impl BuildHasher for U32Hasher {
    type Hasher = Self;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        self.clone()
    }
}

impl Hasher for U32Hasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        debug_assert!(bytes.len() == 4, "for U32Hasher invalid bytes:{:?}", bytes);
        self.key = u32::from_le_bytes(bytes.try_into().unwrap());
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.key.into()
    }
}
