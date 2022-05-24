//! Package implement Concurrent hash map.
//!
//! Quoting from [Wikipedia][pds]:
//!
//! > A data structure is *partially persistent* if all versions can be
//! > accessed but only the newest version can be modified. The data
//! > structure is *fully persistent* if every version can be both accessed
//! > and modified. If there is also a meld or merge operation that can
//! > create a new version from two previous versions, the data structure is
//! > called *confluently persistent*. Structures that are not persistent are
//! > called *ephemeral* data structures.
//!
//! This implementation of hash map cannot be strictly classified into either
//! of the above definition. It supports concurrent writes, using atomic
//! ``Load``, ``Store`` and ``Cas`` operations under the hood, and _does not_
//! provide point in time snapshot for transactional operations or iterative operations.
//!
//! If point in time snapshots are needed refer to [ppom] package, that
//! implement ordered map with multi-reader concurrency and serialised writes.
//!
//! - Each entry in [Map] instance correspond to a {Key, Value} pair.
//! - Parametrised over `key-type` and `value-type`.
//! - API - set(), get(), remove() using key.
//! - Uses ownership model and borrow semantics to ensure safety.
//! - Implement a custom epoch-based-garbage-collection to handle write
//!   concurrency and memory optimization.
//! - No Durability guarantee.
//! - Thread safe for both concurrent writes and concurrent reads.
//!
//! Thread safety
//! -------------
//!
//! As an optimization, creation of a [Map] instance requires the application
//! to supply the level concurrency upfront. That is, if Map is going to be
//! cloned N times for the life-time of the map, then concurrency level is N.
//! Subsequently all access to the internal hash-map is serialized via atomic
//! ``Load``, ``Store`` and ``Cas`` operations.
//!
//! Ownership and Cloning
//! ---------------------
//!
//! Once Map instance is created, cloning is a cheap operation. Maps cloned
//! from the same instances share their hash-trie data structure among
//! its clones. Ownership rules adhere to rust ownership model and all access
//! require mutable reference.
//!
//! Application defined hashing
//! ===========================
//!
//! Parametrise [Map] type with ``H`` for application defined [BuildHasher].
//! This allows interesting and efficient hash-generation for application
//! specific key-set.
//!
//! This package define two off-the-self types implementing BuildHasher.
//!
//! * [U32Hasher], for applications that are going to use u32 as key type
//!   and can guarantee unique keys (that is no collision guarantee).
//! * [DefaultHasher], as default hasher that internally uses google's
//!   city-hash via [fasthash][fasthash] package, this might change in future
//!   releases.
//!
//! [pds]: https://en.wikipedia.org/wiki/Persistent_data_structure
//! [ppom]: https://github.com/bnclabs/cmap
//! [fasthash]: https://github.com/flier/rust-fasthash

#[allow(unused_imports)]
use std::hash::BuildHasher;
use std::{error, fmt, result};

/// Short form to compose Error values.
///
/// Here are few possible ways:
///
/// ```ignore
/// use crate::Error;
/// err_at!(ParseError, msg: "bad argument");
/// ```
///
/// ```ignore
/// use crate::Error;
/// err_at!(ParseError, std::io::read(buf));
/// ```
///
/// ```ignore
/// use crate::Error;
/// err_at!(ParseError, std::fs::read(file_path), "read failed");
/// ```
///
#[macro_export]
macro_rules! err_at {
    ($v:ident, msg: $($arg:expr),+) => {{
        let prefix = format!("{}:{}", file!(), line!());
        Err(Error::$v(prefix, format!($($arg),+)))
    }};
    ($v:ident, $e:expr) => {{
        match $e {
            Ok(val) => Ok(val),
            Err(err) => {
                let prefix = format!("{}:{}", file!(), line!());
                Err(Error::$v(prefix, format!("{}", err)))
            }
        }
    }};
    ($v:ident, $e:expr, $($arg:expr),+) => {{
        match $e {
            Ok(val) => Ok(val),
            Err(err) => {
                let prefix = format!("{}:{}", file!(), line!());
                let msg = format!($($arg),+);
                Err(Error::$v(prefix, format!("{} {}", err, msg)))
            }
        }
    }};
}

/// Error variants that can be returned by this package's API.
///
/// Each variant carries a prefix, typically identifying the
/// error location.
pub enum Error {
    Fatal(String, String),
    GcFail(String, String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        use Error::*;

        match self {
            Fatal(p, msg) => write!(f, "{} Fatal: {}", p, msg),
            GcFail(p, msg) => write!(f, "{} GcFail: {}", p, msg),
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        write!(f, "{}", self)
    }
}

impl error::Error for Error {}

// mod entry;
pub mod gc;
mod hasher;
pub mod map;

pub use hasher::{DefaultHasher, U32Hasher};
pub use map::Map;

/// Type alias for Result return type, used by this package.
pub type Result<T> = result::Result<T, Error>;
