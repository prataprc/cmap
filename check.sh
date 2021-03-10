#! /usr/bin/env bash

export RUST_BACKTRACE=full
export RUSTFLAGS=-g
exec > check.out
exec 2>&1

set -o xtrace

exec_prg() {
    for i in {0..10};
    do
        date; time cargo +nightly test --release -- --nocapture || exit $?
        date; time cargo +nightly test -- --nocapture || exit $?
        date; time cargo +nightly test --features=compact --release -- --nocapture || exit $?
        date; time cargo +nightly test --features=compact -- --nocapture || exit $?
        date; time cargo +nightly bench -- --nocapture || exit $?
        date; time cargo +nightly bench --features=compact -- --nocapture || exit $?
        # repeat this for stable, once package is ready for stable.
    done
}

exec_prg
