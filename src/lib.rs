/*!
Rust implementation of [Pruning Radix Trie, orignially by Wolf Garbe](https://github.com/wolfgarbe/PruningRadixTrie).

A Radix Trie or Patricia Trie is a space-optimized trie (prefix tree).
A Pruning Radix trie is a novel Radix trie algorithm, that allows pruning of the Radix trie and early termination of the lookup.

The trie can only be appended to, terms cannot be removed nor payloads altered.

# Usage
```rust
use pruning_radix_trie::{PruningRadixTrie, Result};

let mut trie = PruningRadixTrie::new();
trie.add("heyo", vec![1, 2, 3], 5);
trie.add("hello", vec![3, 4, 5], 10);
trie.add("hej", vec![5, 6, 7], 20);

let results = trie.find("he", 10);
for Result { term, payload, weight } in results {
    println!("{:10}{:?}{:>4}", term, payload, weight);
}
//hej       [7, 8, 9]  20
//hello     [4, 5, 6]  10
//heyo      [1, 2, 3]   5

let results = trie.find_with_filter("he", 10, |p| p.contains(&3));
for Result { term, payload, weight } in results {
    println!("{:10}{:?}{:>4}", term, payload, weight);
}
//hello     [3, 4, 5]  10
//heyo      [1, 2, 3]   5
```
*/

use std::cmp::Ordering::{Equal, Greater, Less};
use std::cmp::{max, min};
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::ops::Add;

use compact_str::CompactString;

#[derive(Copy, Clone)]
struct NodeId(usize);
type NodeKey = CompactString;
struct Node<T, U>
where
    U: Ord + Copy + Add<Output = U> + Debug,
{
    children: Option<Vec<(NodeKey, NodeId)>>,
    payload: Option<T>,
    weight: Option<U>,
    child_max_weight: Option<U>,
}

pub struct PruningRadixTrie<T, U>
where
    U: Ord + Copy + Add<Output = U> + Debug,
{
    nodes: Vec<Node<T, U>>,
    term_count: usize,
}

#[derive(Debug)]
pub struct Result<'a, T, U>
where
    U: Ord + Copy + Add<Output = U> + Debug,
{
    pub term: NodeKey,
    pub payload: &'a T,
    pub weight: &'a U,
}

impl<'a, T, U> Result<'a, T, U>
where
    U: Ord + Copy + Add<Output = U> + Debug,
{
    pub fn new(term: NodeKey, payload: &'a T, weight: &'a U) -> Self {
        Self {
            term,
            payload,
            weight,
        }
    }
}

struct NodeMatchContext<'a> {
    node_id: NodeId,
    node_index: usize,
    common: usize,
    key: &'a str,
}

enum NodeMatch<'a> {
    NoMatch,
    Equal(NodeMatchContext<'a>),
    IsShorter(NodeMatchContext<'a>),
    IsLonger(NodeMatchContext<'a>),
    CommonSubstring(NodeMatchContext<'a>),
}

impl<'a, T, U> Ord for Result<'a, T, U>
where
    U: Ord + Copy + Add<Output = U> + Debug,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.weight.cmp(self.weight)
    }
}
impl<'a, T, U> PartialOrd for Result<'a, T, U>
where
    U: Ord + Copy + Add<Output = U> + Debug,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.weight.partial_cmp(self.weight)
    }
}

impl<'a, T, U> PartialEq for Result<'a, T, U>
where
    U: Ord + Copy + Add<Output = U> + Debug,
{
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<'a, T, U> Eq for Result<'a, T, U> where U: Ord + Copy + Add<Output = U> + Debug {}

impl<T, U> Default for PruningRadixTrie<T, U>
where
    U: Ord + Copy + Add<Output = U> + Debug,
{
    fn default() -> Self {
        PruningRadixTrie::new()
    }
}

impl<T, U> PruningRadixTrie<T, U>
where
    U: Ord + Copy + Add<Output = U> + Debug,
{
    pub fn new() -> Self {
        PruningRadixTrie {
            nodes: vec![Node {
                children: None,
                payload: None,
                weight: None,
                child_max_weight: None,
            }],
            term_count: 0,
        }
    }

    fn match_children(&self, node_id: NodeId, term: &str) -> NodeMatch {
        if let Some(children) = &self.nodes[node_id.0].children {
            for (index, (key, id)) in children.iter().enumerate() {
                let mut common = 0;
                for i in 0..min(term.len(), key.len()) {
                    if term.as_bytes().get(i).unwrap() == key.as_bytes().get(i).unwrap() {
                        common = i + 1;
                    } else {
                        break;
                    }
                }
                while !term.is_char_boundary(common) {
                    common -= 1;
                }
                if common > 0 {
                    let context = NodeMatchContext {
                        node_id: *id,
                        node_index: index,
                        common,
                        key,
                    };
                    return match (common == term.len(), common == key.len()) {
                        (true, true) => NodeMatch::Equal(context),
                        (true, _) => NodeMatch::IsShorter(context),
                        (_, true) => NodeMatch::IsLonger(context),
                        (_, _) => NodeMatch::CommonSubstring(context),
                    };
                }
            }
        }

        NodeMatch::NoMatch
    }

    fn make_node(
        &mut self,
        children: Option<Vec<(NodeKey, NodeId)>>,
        payload: Option<T>,
        weight: Option<U>,
        child_max_weight: Option<U>,
    ) -> NodeId {
        let node_id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            children,
            payload,
            weight,
            child_max_weight,
        });
        node_id
    }

    pub fn len(&self) -> usize {
        self.term_count
    }

    pub fn is_empty(&self) -> bool {
        self.term_count == 0
    }

    fn append_child<S: Into<NodeKey>>(&mut self, parent_id: NodeId, term: S, child_id: NodeId) {
        let child_node = &self.nodes[child_id.0];
        let insert_index =
            self.get_insert_index(parent_id, child_node.weight, child_node.child_max_weight);

        let parent_node = &mut self.nodes[parent_id.0];
        if let Some(children) = &mut parent_node.children {
            children.insert(insert_index, (term.into(), child_id));
        } else {
            parent_node.children = Some(vec![(term.into(), child_id)]);
        }
    }

    pub fn find<'a>(&'a self, prefix: &str, top_k: usize) -> Vec<Result<'a, T, U>> {
        self.find_with_filter(prefix, top_k, |_| true)
    }

    pub fn find_with_filter<'a>(
        &'a self,
        prefix: &str,
        top_k: usize,
        mut predicate: impl FnMut(&'a T) -> bool,
    ) -> Vec<Result<'a, T, U>> {
        let mut results: BinaryHeap<Result<'a, T, U>> = BinaryHeap::new();
        let mut matched_prefix = NodeKey::default();
        self.find_all_child_terms(
            &self.nodes[0],
            prefix,
            &mut matched_prefix,
            top_k,
            &mut predicate,
            &mut results,
        );
        results.into_sorted_vec()
    }

    fn find_all_child_terms<'a, P>(
        &'a self,
        node: &'a Node<T, U>,
        prefix: &str,
        matched_prefix: &mut NodeKey,
        top_k: usize,
        predicate: &mut P,
        results: &mut BinaryHeap<Result<'a, T, U>>,
    ) where
        P: FnMut(&'a T) -> bool,
    {
        if let Some(children) = &node.children {
            if results.len() == top_k && node.child_max_weight <= results.peek().map(|r| *r.weight)
            {
                return;
            }
            for (term, child_id) in children {
                let child = &self.nodes[child_id.0];

                if results.len() == top_k
                    && child.weight <= results.peek().map(|r| *r.weight)
                    && child.child_max_weight <= results.peek().map(|r| *r.weight)
                {
                    if prefix.is_empty() {
                        continue;
                    } else {
                        break;
                    }
                }

                if prefix.is_empty() || term.starts_with(prefix) {
                    if child.weight.is_some() || node.children.is_some() {
                        matched_prefix.push_str(term);

                        if let (Some(weight), Some(payload)) =
                            (child.weight.as_ref(), child.payload.as_ref())
                        {
                            if predicate(payload) {
                                results.push(Result {
                                    term: matched_prefix.as_str().into(),
                                    payload,
                                    weight,
                                });
                                if results.len() > top_k {
                                    results.pop();
                                }
                            }
                        }
                        self.find_all_child_terms(
                            child,
                            "",
                            matched_prefix,
                            top_k,
                            predicate,
                            results,
                        );
                        matched_prefix.truncate(matched_prefix.len() - term.len());
                    }

                    if !prefix.is_empty() {
                        break;
                    }
                } else if prefix.starts_with(term.as_str()) {
                    matched_prefix.push_str(term);
                    self.find_all_child_terms(
                        child,
                        &prefix[term.len()..],
                        matched_prefix,
                        top_k,
                        predicate,
                        results,
                    );
                    matched_prefix.truncate(matched_prefix.len() - term.len());
                }
            }
        }
    }

    pub fn add(&mut self, term: &str, payload: T, weight: U) {
        let weight = self.add_term(NodeId(0), term, payload, weight);
        self.nodes[0].child_max_weight = max(self.nodes[0].child_max_weight, Some(weight));
    }

    fn get_insert_index(
        &self,
        node_id: NodeId,
        weight: Option<U>,
        child_max_weight: Option<U>,
    ) -> usize {
        if let Some(children) = &self.nodes[node_id.0].children {
            let result = children.binary_search_by(|(_, child_id)| {
                match child_max_weight.cmp(&self.nodes[child_id.0].child_max_weight) {
                    Equal => weight.cmp(&self.nodes[child_id.0].weight),
                    Less => Less,
                    Greater => Greater,
                }
            });
            match result {
                Ok(index) => index,
                Err(index) => index,
            }
        } else {
            0
        }
    }

    fn replace_node(&mut self, parent_id: NodeId, node_index: usize, term: &str, child_id: NodeId) {
        self.nodes[parent_id.0]
            .children
            .as_mut()
            .unwrap()
            .remove(node_index);
        self.append_child(parent_id, term.to_owned(), child_id);
    }

    fn update_child_max_weight(
        &mut self,
        parent_id: NodeId,
        node_id: NodeId,
        node_index: usize,
        new_child_max_weight: U,
    ) {
        let node = &mut self.nodes[node_id.0];
        let new_child_max_weight = Some(new_child_max_weight);
        if node.child_max_weight < new_child_max_weight {
            node.child_max_weight = new_child_max_weight;

            if node_index > 0 {
                let (_, prev_child_id) =
                    self.nodes[parent_id.0].children.as_mut().unwrap()[node_index - 1];
                if node_index > 0
                    || new_child_max_weight > self.nodes[prev_child_id.0].child_max_weight
                {
                    let (term, child_id) = self.nodes[parent_id.0]
                        .children
                        .as_mut()
                        .unwrap()
                        .remove(node_index);
                    self.append_child(parent_id, term, child_id);
                }
            }
        }
    }

    fn add_term(&mut self, curr_id: NodeId, term: &str, payload: T, weight: U) -> U {
        match self.match_children(curr_id, term) {
            NodeMatch::Equal(NodeMatchContext { node_id, .. }) => {
                let node = &mut self.nodes[node_id.0];
                if let Some(node_weight) = node.weight {
                    let new_weight = node_weight + weight;
                    node.weight = Some(new_weight);
                    new_weight
                } else {
                    self.term_count += 1;
                    node.weight = Some(weight);
                    node.payload = Some(payload);
                    weight
                }
            }

            NodeMatch::IsShorter(NodeMatchContext {
                node_id,
                common,
                node_index,
                key,
            }) => {
                let node = &self.nodes[node_id.0];
                let child_id = self.make_node(
                    Some(vec![(key[common..].into(), node_id)]),
                    Some(payload),
                    Some(weight),
                    max(node.weight, node.child_max_weight),
                );

                self.replace_node(curr_id, node_index, &term[0..common], child_id);
                self.term_count += 1;
                weight
            }

            NodeMatch::IsLonger(NodeMatchContext {
                node_id,
                common,
                node_index,
                ..
            }) => {
                let weight = self.add_term(node_id, &term[common..], payload, weight);
                self.update_child_max_weight(curr_id, node_id, node_index, weight);
                weight
            }

            NodeMatch::CommonSubstring(NodeMatchContext {
                node_id,
                common,
                node_index,
                key,
            }) => {
                let node = &self.nodes[node_id.0];
                let key = key[common..].into();
                let child_max_weight = max(node.child_max_weight, max(node.weight, Some(weight)));
                let new_node_id = self.make_node(None, Some(payload), Some(weight), None);
                let child_id = self.make_node(
                    Some(vec![(key, node_id), (term[common..].into(), new_node_id)]),
                    None,
                    None,
                    child_max_weight,
                );

                self.replace_node(curr_id, node_index, &term[0..common], child_id);
                self.term_count += 1;
                weight
            }

            NodeMatch::NoMatch => {
                let node_id = self.make_node(None, Some(payload), Some(weight), Default::default());
                self.append_child(curr_id, term.to_owned(), node_id);
                self.term_count += 1;
                weight
            }
        }
    }

    // diagnostic code
    #[allow(dead_code)]
    fn dump_node(&self, node_id: NodeId, term: &str, len: usize) {
        let parent = &self.nodes[node_id.0];
        println!(
            "{:->len$} ({:?}/{:?}) id: {}, {} children, payload: {}",
            term,
            parent.weight,
            parent.child_max_weight,
            node_id.0,
            self.nodes[node_id.0]
                .children
                .as_ref()
                .map_or(0, |c| c.len()),
            if parent.payload.is_some() {
                "yes"
            } else {
                "no"
            }
        );
        if let Some(children) = &self.nodes[node_id.0].children {
            for (term, child_id) in children {
                self.dump_node(*child_id, term, len + term.len());
            }
        }
    }

    #[allow(dead_code)]
    fn dump(&self) {
        self.dump_node(NodeId(0), "<root>", 0);
    }

    #[allow(dead_code)]
    fn assert_child_order(&self, node_id: NodeId) {
        if let Some(children) = &self.nodes[node_id.0].children {
            assert!(!children.is_empty());
            let mut last_child_max_weight = None;
            for (_, child_id) in children {
                let child_max_weight = self.nodes[child_id.0].child_max_weight;
                if let Some(last_child_max_weight) = last_child_max_weight {
                    assert!(child_max_weight <= last_child_max_weight);
                } else {
                    last_child_max_weight = Some(child_max_weight);
                }
                self.assert_child_order(*child_id);
            }
        }
    }

    #[allow(dead_code)]
    fn assert_child_max(&self, node_id: NodeId, child_max_weight: Option<U>) {
        if let Some(children) = &self.nodes[node_id.0].children {
            for (_, child_id) in children {
                let child = &self.nodes[child_id.0];
                assert_eq!(child.weight.is_some(), child.payload.is_some());
                assert!(child.weight <= child_max_weight);
                assert!(child.child_max_weight <= child_max_weight);

                self.assert_child_max(*child_id, child.child_max_weight);
            }
        }
    }

    #[allow(dead_code)]
    fn assert_invariants(&self) {
        self.assert_child_order(NodeId(0));
        self.assert_child_max(NodeId(0), self.nodes[0].child_max_weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let trie: PruningRadixTrie<(), u32> = Default::default();
        assert!(trie.is_empty());
        let results = trie.find("hello", 10);
        assert!(results.is_empty());
    }
    #[test]
    fn single_term() {
        let mut trie = PruningRadixTrie::new();
        trie.add("hello", (), 10);

        assert_eq!(trie.len(), 1);

        trie.assert_invariants();

        let results = trie.find("h", 1);
        assert!(!results.is_empty());
        assert_eq!(results[0], Result::new("hello".into(), &(), &10_u32));
    }

    // (true, true) => NodeMatch::Equal(context),
    // (true, _) => NodeMatch::IsShorter(context),
    // (_, true) => NodeMatch::IsLonger(context),
    // (_, _) => NodeMatch::CommonSubstring(context),
    // NoMatch

    #[test]
    fn equal_match_same_term() {
        let mut trie = PruningRadixTrie::new();
        trie.add("hello", (), 10);
        trie.add("hello", (), 10);

        assert_eq!(trie.len(), 1);

        trie.assert_invariants();

        let results = trie.find("h", 1);
        assert!(!results.is_empty());
        assert_eq!(results[0], Result::new("hello".into(), &(), &20_u32));
    }

    #[test]
    fn equal_match_same_node() {
        let mut trie = PruningRadixTrie::new();
        trie.add("heyo", (), 5);
        trie.add("hello", (), 10);
        trie.add("he", (), 20);

        assert_eq!(trie.len(), 3);

        trie.assert_invariants();
        trie.dump();

        let results = trie.find("he", 3);
        assert_eq!(
            &results,
            &vec![
                Result::new("he".into(), &(), &20_u32),
                Result::new("hello".into(), &(), &10_u32),
                Result::new("heyo".into(), &(), &5_u32)
            ]
        );
    }
    #[test]
    fn test_char_boundary() {
        let mut trie = PruningRadixTrie::new();
        trie.add("hello 🙂", (), 10);
        trie.add("hello 😊", (), 20);

        assert_eq!(trie.len(), 2);

        let results = trie.find("he", 3);
        assert_eq!(
            &results,
            &vec![
                Result::new("hello 😊".into(), &(), &20_u32),
                Result::new("hello 🙂".into(), &(), &10_u32),
            ]
        );
    }

    #[test]
    fn same_rank() {
        let mut trie = PruningRadixTrie::new();
        trie.add("hello world", (), 10);
        trie.add("hello you", (), 10);

        assert_eq!(trie.len(), 2);

        let results = trie.find("hello", 3);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn i64_rank() {
        let mut trie = PruningRadixTrie::new();
        trie.add("hello world", (), 10_i64);
        trie.add("hello you", (), 20_i64);
        trie.add("hello long", (), 50_i64);

        assert_eq!(trie.len(), 3);

        let results = trie.find("hello", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn predicate() {
        let mut trie = PruningRadixTrie::new();
        trie.add("hello world", true, 10);
        trie.add("hello you", false, 20);
        trie.add("hello long", false, 50);

        let results = trie.find_with_filter("hello", 3, |p| *p);

        assert_eq!(
            &results,
            &vec![Result::new("hello world".into(), &true, &10)]
        );
    }

    #[test]
    fn predicate_mutate_state() {
        let mut trie = PruningRadixTrie::new();
        trie.add("hello world", true, 10);
        trie.add("hello you", false, 20);
        trie.add("hello long", false, 50);

        let mut comparisons = 0;
        let results = trie.find_with_filter("hello", 3, |p| {
            comparisons += 1;
            *p
        });
        assert_eq!(
            &results,
            &vec![Result::new("hello world".into(), &true, &10)]
        );
        assert!(comparisons > 0);
    }

    #[test]
    fn complex_1() {
        let mut trie = PruningRadixTrie::new();
        trie.add("hello", (), 12);
        trie.add("he", (), 15);
        trie.add("helloworld", (), 5);
        trie.add("hellowood", (), 20);
        trie.add("hella", (), 30);
        trie.add("helli", (), 21);
        trie.add("hellia", (), 10);
        trie.add("goodbye", (), 40);
        assert_eq!(trie.len(), 8);

        trie.dump();
        trie.assert_invariants();

        let results = trie.find("h", 5);
        assert_eq!(
            &results,
            &vec![
                Result::new("hella".into(), &(), &30_u32),
                Result::new("helli".into(), &(), &21_u32),
                Result::new("hellowood".into(), &(), &20_u32),
                Result::new("he".into(), &(), &15_u32),
                Result::new("hello".into(), &(), &12_u32)
            ]
        );

        let results = trie.find("hellowo", 5);
        assert_eq!(
            &results,
            &vec![
                Result::new("hellowood".into(), &(), &20_u32),
                Result::new("helloworld".into(), &(), &5_u32),
            ]
        );
    }

    #[test]
    fn example() {
        let mut trie = PruningRadixTrie::new();
        trie.add("heyo", vec![1, 2, 3], 5);
        trie.add("hello", vec![3, 4, 5], 10);
        trie.add("hej", vec![5, 6, 7], 20);

        let results = trie.find("he", 10);

        for Result {
            term,
            payload,
            weight,
        } in results
        {
            println!("{:10}{:?}{:>4}", term, payload, weight);
        }
        //hej       [7, 8, 9]  20
        //hello     [4, 5, 6]  10
        //heyo      [1, 2, 3]   5

        let results = trie.find_with_filter("he", 10, |p| p.contains(&3));
        for Result {
            term,
            payload,
            weight,
        } in results
        {
            println!("{:10}{:?}{:>4}", term, payload, weight);
        }
        //hello     [3, 4, 5]  10
        //heyo      [1, 2, 3]   5
    }

    #[test]
    #[ignore]
    fn terms_list() {
        use std::fs::File;
        use std::io::prelude::*;
        use std::io::BufReader;

        fn load_terms(trie: &mut PruningRadixTrie<(), u32>) {
            let terms_file = File::open("terms.txt").expect("file not found!");

            let buf_reader = BufReader::new(terms_file);

            for line in buf_reader.lines().flatten() {
                if let Some((term, freq)) = line.split_once('\t') {
                    if let Ok(freq) = freq.parse::<u32>() {
                        trie.add(term, (), freq);
                    }
                }
            }
        }
        let mut trie = PruningRadixTrie::new();
        let top_k = 20;
        load_terms(&mut trie);
        let results = trie.find("wik", top_k);

        for Result { term, weight, .. } in &results {
            println!("{:30}{:>7}", term, weight);
        }

        assert_eq!(results.len(), top_k);
        assert!(is_descending(&results));
    }

    fn is_descending<T, U>(results: &[Result<T, U>]) -> bool
    where
        U: Ord + Copy + Add<Output = U> + Debug,
    {
        results.windows(2).all(|w| *w[1].weight <= *w[0].weight)
    }
}
