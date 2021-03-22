* `get_unchecked()` optimization on Vec type, for better performance.
* Implement Dict type as a single-threaded variant of Map, using the
  same data structure primitives but optimized for single threaded
  set(), get(), remove().
* Performance bechmark Dict with SwissTable and std::collections.
* Implement Maps type that internally maintains a `N` Map shards.
  Write test cases and check for performance improvements.
* Try to avoid Clone trait contraint for K and V type-parameter.
* optimize create() API to avoid copy/clone, ends up a waste if
  key alread present.
