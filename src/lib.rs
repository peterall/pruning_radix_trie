pub mod pruning_radix_trie {
    use std::cmp::Ordering::{Equal, Greater, Less};
    use std::cmp::{max, min};

    #[derive(Copy, Clone)]
    struct NodeId(usize);
    struct Node {
        children: Option<Vec<(String, NodeId)>>,
        weight: u32,
        child_max_weight: u32,
    }

    pub struct PruningRadixTrie {
        nodes: Vec<Node>,
        term_count: usize,
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

    impl Default for PruningRadixTrie {
        fn default() -> Self {
            PruningRadixTrie::new()
        }
    }

    impl PruningRadixTrie {
        pub fn new() -> Self {
            PruningRadixTrie {
                nodes: vec![Node {
                    children: None,
                    weight: 0,
                    child_max_weight: 0,
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
            children: Option<Vec<(String, NodeId)>>,
            weight: u32,
            child_max_weight: u32,
        ) -> NodeId {
            let node_id = NodeId(self.nodes.len());
            self.nodes.push(Node {
                children,
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

        fn append_child(&mut self, parent_id: NodeId, term: String, child_id: NodeId) {
            let child_node = &self.nodes[child_id.0];
            let insert_index =
                self.get_insert_index(parent_id, child_node.weight, child_node.child_max_weight);

            let parent_node = &mut self.nodes[parent_id.0];
            if let Some(children) = &mut parent_node.children {
                children.insert(insert_index, (term, child_id));
            } else {
                parent_node.children = Some(vec![(term, child_id)]);
            }
        }

        pub fn find(&self, search_term: &str, top_k: usize) -> Vec<(String, u32)> {
            let mut results = Vec::with_capacity(top_k);
            let mut matched_prefix = String::with_capacity(32);
            self.find_all_child_terms(
                &self.nodes[0],
                search_term,
                &mut matched_prefix,
                top_k,
                &mut results,
            );
            results
        }

        fn add_top_k_result(
            &self,
            term: &str,
            weight: u32,
            top_k: usize,
            results: &mut Vec<(String, u32)>,
        ) {
            if results.len() < top_k || weight >= results[top_k - 1].1 {
                let position = match results.binary_search_by(|r| weight.cmp(&r.1)) {
                    Ok(pos) => pos,
                    Err(pos) => pos,
                };
                if results.len() == top_k {
                    results.remove(top_k - 1);
                }
                results.insert(position, (term.to_owned(), weight));
            }
        }

        fn find_all_child_terms(
            &self,
            node: &Node,
            prefix: &str,
            matched_prefix: &mut String,
            top_k: usize,
            results: &mut Vec<(String, u32)>,
        ) {
            if let Some(children) = &node.children {
                if results.len() == top_k && node.child_max_weight <= results[top_k - 1].1 {
                    return;
                }
                for (term, child_id) in children {
                    let child = &self.nodes[child_id.0];

                    if results.len() == top_k
                        && child.weight <= results[top_k - 1].1
                        && child.child_max_weight <= results[top_k - 1].1
                    {
                        if prefix.is_empty() {
                            continue;
                        } else {
                            break;
                        }
                    }

                    if prefix.is_empty() || term.starts_with(prefix) {
                        if child.weight > 0 || node.children.is_some() {
                            matched_prefix.push_str(term);

                            if child.weight > 0 {
                                self.add_top_k_result(matched_prefix, child.weight, top_k, results);
                            }
                            self.find_all_child_terms(child, "", matched_prefix, top_k, results);
                            matched_prefix.truncate(matched_prefix.len() - term.len());
                        }

                        if !prefix.is_empty() {
                            break;
                        }
                    } else if prefix.starts_with(term) {
                        matched_prefix.push_str(term);
                        self.find_all_child_terms(
                            child,
                            &prefix[term.len()..],
                            matched_prefix,
                            top_k,
                            results,
                        );
                        matched_prefix.truncate(matched_prefix.len() - term.len());
                    }
                }
            }
        }

        pub fn add(&mut self, term: &str, weight: u32) {
            let weight = self.add_term(NodeId(0), term, weight);
            self.nodes[0].child_max_weight = max(weight, self.nodes[0].child_max_weight);
        }

        fn get_insert_index(&self, node_id: NodeId, weight: u32, child_max_weight: u32) -> usize {
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

        fn replace_node(
            &mut self,
            parent_id: NodeId,
            node_index: usize,
            term: &str,
            child_id: NodeId,
        ) {
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
            new_child_max_weight: u32,
        ) {
            let node = &mut self.nodes[node_id.0];
            if node.child_max_weight < new_child_max_weight {
                node.child_max_weight = new_child_max_weight;

                if node_index > 0 {
                    let (_, prev_child_id) =
                        self.nodes[parent_id.0].children.as_mut().unwrap()[node_index - 1];
                    if node_index > 0
                        && new_child_max_weight > self.nodes[prev_child_id.0].child_max_weight
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

        fn add_term(&mut self, curr_id: NodeId, term: &str, weight: u32) -> u32 {
            match self.match_children(curr_id, term) {
                NodeMatch::Equal(NodeMatchContext { node_id, .. }) => {
                    let node = &mut self.nodes[node_id.0];
                    if node.weight == 0 {
                        self.term_count += 1;
                    }
                    node.weight += weight;
                    node.weight
                }

                NodeMatch::IsShorter(NodeMatchContext {
                    node_id,
                    common,
                    node_index,
                    key,
                }) => {
                    let node = &self.nodes[node_id.0];
                    let child_max_weight = max(node.weight, node.child_max_weight);
                    let child_id = self.make_node(
                        Some(vec![(key[common..].to_owned(), node_id)]),
                        weight,
                        child_max_weight,
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
                    let weight = self.add_term(node_id, &term[common..], weight);
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
                    let key = key[common..].to_owned();

                    let child_max_weight = max(node.child_max_weight, max(weight, node.weight));

                    let new_node_id = self.make_node(None, weight, 0);
                    let child_id = self.make_node(
                        Some(vec![
                            (key, node_id),
                            (term[common..].to_owned(), new_node_id),
                        ]),
                        0,
                        child_max_weight,
                    );

                    self.replace_node(curr_id, node_index, &term[0..common], child_id);
                    self.term_count += 1;
                    weight
                }

                NodeMatch::NoMatch => {
                    let node_id = self.make_node(None, weight, 0);
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
                "{:->len$} ({}/{}) id: {}, {} children",
                term,
                parent.weight,
                parent.child_max_weight,
                node_id.0,
                self.nodes[node_id.0]
                    .children
                    .as_ref()
                    .map_or(0, |c| c.len())
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
        fn assert_child_max(&self, node_id: NodeId, child_max_weight: u32) {
            if let Some(children) = &self.nodes[node_id.0].children {
                for (_, child_id) in children {
                    let child = &self.nodes[child_id.0];
                    assert!(child.weight <= child_max_weight);
                    assert!(child.child_max_weight <= child_max_weight);

                    self.assert_child_max(*child_id, child.child_max_weight);
                }
            }
        }

        #[allow(dead_code)]
        fn assert_invariants(&self) {
            self.assert_child_order(NodeId(0));
            self.assert_child_max(NodeId(0), u32::MAX);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn empty() {
            let trie = PruningRadixTrie::default();
            assert!(trie.is_empty());
            let results = trie.find("hello", 10);
            assert!(results.is_empty());
        }
        #[test]
        fn single_term() {
            let mut trie = PruningRadixTrie::new();
            trie.add("hello", 10);

            assert_eq!(trie.len(), 1);

            trie.assert_invariants();

            let results = trie.find("h", 1);
            assert!(!results.is_empty());
            assert_eq!(results[0], ("hello".to_owned(), 10_u32));
        }

        // (true, true) => NodeMatch::Equal(context),
        // (true, _) => NodeMatch::IsShorter(context),
        // (_, true) => NodeMatch::IsLonger(context),
        // (_, _) => NodeMatch::CommonSubstring(context),
        // NoMatch

        #[test]
        fn equal_match_same_term() {
            let mut trie = PruningRadixTrie::new();
            trie.add("hello", 10);
            trie.add("hello", 10);

            assert_eq!(trie.len(), 1);

            trie.assert_invariants();

            let results = trie.find("h", 1);
            assert!(!results.is_empty());
            assert_eq!(results[0], ("hello".to_owned(), 20_u32));
        }

        #[test]
        fn equal_match_same_node() {
            let mut trie = PruningRadixTrie::new();
            trie.add("heyo", 5);
            trie.add("hello", 10);
            trie.add("he", 20);

            assert_eq!(trie.len(), 3);

            trie.assert_invariants();

            let results = trie.find("he", 3);
            assert_eq!(
                &results,
                &vec![
                    ("he".to_owned(), 20_u32),
                    ("hello".to_owned(), 10_u32),
                    ("heyo".to_owned(), 5_u32)
                ]
            );
        }
        #[test]
        fn test_char_boundary() {
            let mut trie = PruningRadixTrie::new();
            trie.add("hello 🙂", 10);
            trie.add("hello 😊", 20);

            assert_eq!(trie.len(), 2);

            let results = trie.find("he", 3);
            assert_eq!(
                &results,
                &vec![
                    ("hello 😊".to_owned(), 20_u32),
                    ("hello 🙂".to_owned(), 10_u32),
                ]
            );
        }

        #[test]
        fn same_rank() {
            let mut trie = PruningRadixTrie::new();
            trie.add("hello world", 10);
            trie.add("hello you", 10);

            assert_eq!(trie.len(), 2);

            let results = trie.find("hello", 3);
            assert_eq!(results.len(), 2);
        }

        #[test]
        fn complex_1() {
            let mut trie = PruningRadixTrie::new();
            trie.add("hello", 12);
            trie.add("he", 15);
            trie.add("helloworld", 5);
            trie.add("hellowood", 20);
            trie.add("hella", 30);
            trie.add("helli", 21);
            trie.add("hellia", 10);
            trie.add("goodbye", 40);
            assert_eq!(trie.len(), 8);

            trie.dump();
            trie.assert_invariants();

            let results = trie.find("h", 5);
            assert_eq!(
                &results,
                &vec![
                    ("hella".to_owned(), 30_u32),
                    ("helli".to_owned(), 21_u32),
                    ("hellowood".to_owned(), 20_u32),
                    ("he".to_owned(), 15_u32),
                    ("hello".to_owned(), 12_u32)
                ]
            );

            let results = trie.find("hellowo", 5);
            assert_eq!(
                &results,
                &vec![
                    ("hellowood".to_owned(), 20_u32),
                    ("helloworld".to_owned(), 5_u32),
                ]
            );
        }

        #[test]
        #[ignore]
        fn terms_list() {
            use std::fs::File;
            use std::io::prelude::*;
            use std::io::BufReader;
            use unidecode::unidecode;

            fn load_terms(trie: &mut PruningRadixTrie) {
                let terms_file = File::open("terms.txt").expect("file not found!");

                let buf_reader = BufReader::new(terms_file);

                for line in buf_reader.lines().flatten() {
                    if let Some((term, freq)) = line.split_once('\t') {
                        if let Ok(freq) = freq.parse::<u32>() {
                            trie.add(&unidecode(term), freq);
                        }
                    }
                }
            }
            let mut trie = PruningRadixTrie::new();

            load_terms(&mut trie);
            let results = trie.find("hello", 10);

            for (term, weight) in &results {
                println!("{:30}{:>7}", term, weight);
            }

            assert_eq!(results.len(), 10);
            assert!(is_descending(&results));
        }

        fn is_descending(results: &[(String, u32)]) -> bool {
            results.windows(2).all(|w| w[1].1 <= w[0].1)
        }
    }
}
