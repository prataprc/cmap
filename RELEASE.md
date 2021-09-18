0.3.0
=====

* use CoW instread of atomic-pointer juggling to handle colliding entries.
* add `get_with()` api.
* improve test cases.
* performance benchmark with flurry.
* bin/perf: measure with high contention rate load.
* bin/perf: use u8/u32 as key type. though needs to be manually changed.
* clippy fixes

0.2.0
=====

* Add performance benchmark for DashMap.
* Makefile and other package management scripts.
* Add ci scripts.
* Improve test cases (randomize gc-period).

Refer to [release-checklist][release-checklist].

[release-checklist]: https://prataprc.github.io/rust-crates-release-checklist.html
