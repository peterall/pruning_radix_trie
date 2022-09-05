# Pruning Radix Trie
Rust implementation of [Pruning Radix Trie by Wolf Garbe](https://github.com/wolfgarbe/PruningRadixTrie) (see credits/PruningRadixTrieLicense.txt).

## Usage
Add terms with payloads to the trie with:
```rust
pub fn add(&mut self, term: &str, payload: T, weight: U);
```
After which you can prefix match with:
```rust
pub fn find(&self, prefix: &str, top_k: usize) 
    -> Vec<(String, &T, &U)>
```
Results are returned in descending order based on weight.

## Example
```rust
use pruning_radix_trie::PruningRadixTrie;

fn main() {
    let mut trie = PruningRadixTrie::new();
    trie.add("heyo", vec![1, 2, 3], 5);
    trie.add("hello", vec![4, 5, 6], 10);
    trie.add("hej", vec![7, 8, 9], 20);

    let results = trie.find("he", 10);

    for (term, payload, weight) in results {
        println!("{:10}{:?}{:>4}", term, payload, weight);
    }
    //hej       [7, 8, 9]  20
    //hello     [4, 5, 6]  10
    //heyo      [1, 2, 3]   5
}
```
## Testing
### Measuring code coverage

N.B.: At the moment, the nightly channel of Rust is required.

First of all, install grcov
```sh
cargo install grcov
```

Second, install the llvm-tools Rust component (`llvm-tools-preview` for now, it might become `llvm-tools` soon):
```sh
rustup component add llvm-tools-preview
```
To run tests with code coverage run
```
bash run_source_cov.sh
```
A html coverage report will be generated in `./target/debug/coverage/`

See [rust-code-coverage-sample](https://github.com/marco-c/rust-code-coverage-sample) for details.