# Package not ready for stable.

build:
	# ... build ...
	cargo +nightly build
	cargo +nightly build --features=compact
	# ... test ...
	cargo +nightly test --no-run
	cargo +nightly test --features=compact --no-run
	# ... bench ...
	cargo +nightly bench --no-run
	# ... doc ...
	cargo +nightly doc
	# ... bins ...
	cargo +nightly build --release --bin perf --features=perf
	cargo +nightly build --release --bin perf --features=perf,compact
	# ... meta commands ...
	cargo +nightly clippy --all-targets --all-features
flamegraph:
	cargo flamegraph --features=perf --release --bin=perf -- --loads 10000000 --gets 10000000 --threads 16
prepare:
	check.sh
	perf.sh
clean:
	cargo clean
	rm -f check.out perf.out flamegraph.svg perf.data perf.data.old
