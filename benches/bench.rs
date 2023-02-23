#[macro_use]
extern crate lazy_static;
extern crate criterion;
extern crate pruning_radix_trie;

use criterion::{criterion_group, criterion_main, Criterion};
use pruning_radix_trie::PruningRadixTrie;
use std::fs;
use std::path::Path;

lazy_static! {
    static ref TRIE: PruningRadixTrie<(), i32> = {
        let path = Path::new("/tmp/data/benchmark/weighted_strings.txt");
        let contens: String = fs::read_to_string(path).unwrap();
        let mut trie = PruningRadixTrie::new();
        for line in contens.lines() {
            let line_splitted: Vec<&str> = line.split('\t').collect();
            let string = line_splitted[0];
            let weight = line_splitted[1].parse::<i32>().unwrap();
            trie.add(string, (), weight);
        }
        trie
    };
}

fn insert() {
    // Note: to get a benchmark data
    // wget https://gist.githubusercontent.com/subpath/c19778c9549e5dde02a405dd97fa7014/raw/6fe9433996607be9ceca6dc29e1d88582d64f5d1/weighted_strings.txt -P /tmp/data/benchmark
    let path = Path::new("/tmp/data/benchmark/weighted_strings.txt");
    let contens: String = fs::read_to_string(path).unwrap();
    let mut trie = PruningRadixTrie::new();
    for line in contens.lines() {
        let line_splitted: Vec<&str> = line.split('\t').collect();
        let string = line_splitted[0];
        let weight = line_splitted[1].parse::<i32>().unwrap();
        trie.add(string, (), weight);
    }
}

fn lookup_100() {
    let k = 100;
    TRIE.find("pi", k);
    TRIE.find("pis", k);
    TRIE.find("p", k);
    TRIE.find("pineapple", k);
}

fn lookup_1000() {
    let k = 1000;
    TRIE.find("pi", k);
    TRIE.find("pis", k);
    TRIE.find("p", k);
    TRIE.find("pineapple", k);
}

fn lookup_max() {
    let k = usize::MAX;
    TRIE.find("pi", k);
    TRIE.find("pis", k);
    TRIE.find("p", k);
    TRIE.find("pineapple", k);
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pruning_radix_trie");
    group.sample_size(10);
    group.bench_function("insert", |b| b.iter(insert));
    group.bench_function("lookup_100", |b| b.iter(lookup_100));
    group.bench_function("lookup_1000", |b| b.iter(lookup_1000));
    group.bench_function("lookup_max", |b| b.iter(lookup_max));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
