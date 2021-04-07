#! /usr/bin/env bash

exec > perf.out
exec 2>&1

set -o xtrace

PERF=$HOME/.cargo/target/release/perf

date; time cargo +nightly bench -- --nocapture || exit $?
date; time cargo +nightly bench --features=compact -- --nocapture || exit $?

date; time cargo +nightly run --release --bin perf --features=perf -- --loads 10000000 --gets 10000000 --sets 500000 --rems 500000 --threads 8 || exit $?
date; valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $PERF --loads 1000000 --gets 1000000 --sets 50000 --rems 50000 || exit $?

date; time cargo +nightly run --release --bin perf --features=perf,compact -- --loads 10000000 --gets 10000000 --sets 500000 --rems 500000 --threads 8 || exit $?
date; valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $PERF --loads 1000000 --gets 1000000 --sets 50000 --rems 50000 || exit $?
