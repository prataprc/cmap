# Package not ready for stable.

build:
	# ... build ...
	cargo +nightly build
	# cargo +stable build
	cargo +nightly build --features=compact
	# cargo +stable build --features=compact
	# ... test ...
	cargo +nightly test --no-run
	# cargo +stable test --no-run
	cargo +nightly test --features=compact --no-run
	# cargo +stable test --features=compact --no-run
	# ... bench ...
	cargo +nightly bench --no-run
	# cargo +stable bench --no-run
	# ... bins ...
	cargo +nightly build --release --bin perf --features=perf
	# cargo +stable build --release --bin perf --features=perf
	cargo +nightly build --release --bin perf --features=perf,compact
	# cargo +stable build --release --bin perf --features=perf,compact
	# ... doc ...
	cargo +nightly doc
	# cargo +stable doc
	# ... meta commands ...
	cargo +nightly clippy --all-targets --all-features
flamegraph:
	cargo flamegraph --features=perf --release --bin=perf -- --loads 10000000 --gets 10000000 --threads 16
prepare:
	check.sh
	perf.sh
clean:
	rm -f check.out perf.out flamegraph.svg perf.data perf.data.old
