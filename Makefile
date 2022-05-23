build:
	# ... build ...
	cargo +stable build
	cargo +stable build --features=compact
	cargo +nightly build
	cargo +nightly build --features=compact
	#
	# ... test ...
	cargo +stable test --no-run
	cargo +stable test --features=compact --no-run
	cargo +nightly test --no-run
	cargo +nightly test --features=compact --no-run
	#
	# ... bench ...
	cargo +stable bench --no-run
	cargo +nightly bench --no-run
	#
	# ... doc ...
	cargo +stable doc
	cargo +nightly doc
	#
	# ... bins ...
	cargo +stable build --release --bin perf --features=perf
	cargo +stable build --release --bin perf --features=perf,compact
	cargo +nightly build --release --bin perf --features=perf
	cargo +nightly build --release --bin perf --features=perf,compact
	#
	# ... meta commands ...
	cargo +nightly clippy --all-targets --all-features

test:
	# ... test ...
	cargo +stable test
	cargo +nightly test

bench:
	# ... bench ...
	cargo +stable bench
	cargo +nightly bench

flamegraph:
	cargo flamegraph --features=perf --release --bin=perf -- --loads 10000000 --gets 10000000 --threads 16

prepare: build test bench
	check.sh check.out
	perf.sh perf.out

clean:
	cargo clean
	rm -f check.out perf.out flamegraph.svg perf.data perf.data.old
