* `get_unchecked()` optimization on Vec type, for better performance.
* Implement Dict type as a single-threaded variant of Map, using the
  same data structure primitives but optimized for single threaded
  set(), get(), remove().
* Try to avoid Clone trait contraint for K and V type-parameter.
