#!/bin/sh

rm -rf ./target *.prof*

export RUSTFLAGS="-Zinstrument-coverage"

export LLVM_PROFILE_FILE="pruninx-radix-trie-%p-%m.profraw"

cargo build

cargo test

grcov . --binary-path ./target/debug -s . -t html --branch --ignore-not-existing -o ./target/debug/coverage/