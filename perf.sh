PERF=$HOME/.cargo/target/release/perf

date; cargo run --release --bin perf --features=perf -- --loads 10000000 --gets 10000000 --sets 500000 --rems 500000
echo ".......... VALGRIND on perf ..............."
date; valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $PERF --loads 1000000 --gets 1000000 --sets 50000 --rems 50000 --threads 1 > valgrind.perf 2>&1

date; cargo run --release --bin perf --features=perf,compact -- --loads 10000000 --gets 10000000 --sets 500000 --rems 500000
echo ".......... VALGRIND on perf with compact ..............."
date; valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $PERF --loads 1000000 --gets 1000000 --sets 50000 --rems 50000 --threads 1 > valgrind.perf 2>&1
