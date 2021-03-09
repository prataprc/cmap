# Package not ready for stable.

build:
	cargo +nightly build
	# cargo +stable build
	cargo +nightly build --features=compact
	# cargo +stable build --features=compact
	cargo +nightly test --no-run
	# cargo +stable test --no-run
	cargo +nightly test --features=compact --no-run
	# cargo +stable test --features=compact --no-run
	cargo +nightly build --release --bin perf --features=perf
	# cargo +stable build --release --bin perf --features=perf
	cargo +nightly build --release --bin perf --features=perf,compact
	# cargo +stable build --release --bin perf --features=perf,compact
	cargo +nightly doc
	# cargo +stable doc
prepare:
	check.sh
	perf.sh
clean:
	rm -f check.out perf.out
