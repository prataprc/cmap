build:
	cargo build
	cargo build --features=compact
	cargo test --no-run
	cargo test --features=compact --no-run
	cargo build --release --bin perf --features=perf
	cargo build --release --bin perf --features=perf,compact
	cargo doc
prepare:
	check.sh
	perf.sh
clean:
	rm -f a b c d e f g h i j k l m n o p q r s t u v w x y z out.test_map out.arr_map out.dash_map valgrind.perf
