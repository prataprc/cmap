# Package not ready for stable.

build:
	# ... build ...
	# TODO: cargo +stable build
	# TODO: cargo +stable build --features=compact
	cargo +nightly build
	cargo +nightly build --features=compact
	#
	# ... test ...
	# TODO: cargo +stable test --no-run
	# TODO: cargo +stable test --features=compact --no-run
	cargo +nightly test --no-run
	cargo +nightly test --features=compact --no-run
	#
	# ... bench ...
	# TODO: cargo +stable bench --no-run
	cargo +nightly bench --no-run
	#
	# ... doc ...
	# TODO: cargo +stable doc
	cargo +nightly doc
	#
	# ... bins ...
	# TODO: cargo +stable build --release --bin perf --features=perf
	# TODO: cargo +stable build --release --bin perf --features=perf,compact
	cargo +nightly build --release --bin perf --features=perf
	cargo +nightly build --release --bin perf --features=perf,compact
	#
	# ... meta commands ...
	cargo +nightly clippy --all-targets --all-features

test:
	# ... test ...
	# TODO: cargo +stable test
	cargo +nightly test

bench:
	# ... bench ...
	# TODO: cargo +stable bench
	cargo +nightly bench

flamegraph:
	cargo flamegraph --features=perf --release --bin=perf -- --loads 10000000 --gets 10000000 --threads 16

prepare: build test bench
	check.sh check.out
	perf.sh perf.out

clean:
	cargo clean
	rm -f check.out perf.out flamegraph.svg perf.data perf.data.old
