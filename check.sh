#! /usr/bin/env bash

exec > check.out
exec 2>&1

set -o xtrace

exec_prg() {
    date;
    for i in {0..10};
    do
        time cargo +nightly test --release -- --nocapture >> check.out || exit $?
        time cargo +nightly test -- --nocapture >> check.out || exit $?
        time cargo +nightly test --features=compact --release -- --nocapture >> check.out || exit $?
        time cargo +nightly test --features=compact -- --nocapture >> check.out || exit $?
        # repeat this for stable, once package is ready for stable.
    done
}

exec_prg
