#! /usr/bin/env bash

echo "out.test_map"
time cargo test --release test_map -- --nocapture > out.test_map || exit $?
echo
for i in a b c d e f g h i j k l o p q r s t u v w x y z; do
    echo $i
    time cargo test --release test_map -- --nocapture  > $i || exit $?
    echo
done

echo "out.arr_map"
time cargo test --release arr_map -- --nocapture > out.arr_map || exit $?
echo
for i in a b c d e f g h i j k l o p q r s t u v w x y z; do
    echo $i
    time cargo test --release arr_map -- --nocapture  > $i || exit $?
    echo
done

echo "out.dash_map"
time cargo test --release dash_map -- --nocapture > out.dash_map || exit $?
echo
for i in a b c d e f g h i j k l o p q r s t u v w x y z; do
    echo $i
    time cargo test --release dash_map -- --nocapture  > $i || exit $?
    echo
done
