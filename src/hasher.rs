// Module provies hasher implementation for generating key digest.
//
// * **[DefaultHasher]**, wraps google's city hash for using it with [crate::Map]
// * **[U32Hasher]**, can be used for [u32] type keys, where keys are directly
//   returned as the hash digest.

use fasthash::{
    self,
    city::crc::{Hash128, Hasher128},
};

use std::{
    convert::TryInto,
    hash::{BuildHasher, Hasher},
};

/// Type uses google's city hash to convert [Hash]able key into ``u32``.
/// Refer [fasthash] for details.
pub struct DefaultHasher {
    hash_builder: Hash128,
}

impl Default for DefaultHasher {
    fn default() -> Self {
        DefaultHasher::new()
    }
}

impl DefaultHasher {
    pub fn new() -> DefaultHasher {
        DefaultHasher {
            hash_builder: Hash128,
        }
    }
}

impl Clone for DefaultHasher {
    #[inline]
    fn clone(&self) -> Self {
        DefaultHasher {
            hash_builder: Hash128,
        }
    }
}

impl BuildHasher for DefaultHasher {
    type Hasher = Hasher128;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        self.hash_builder.build_hasher()
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
