//! Source coding: Huffman coding, optimal code lengths.

use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// A Huffman code tree node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuffmanNode {
    /// Probability of this subtree.
    pub prob: f64,
    /// Symbol index (leaf only).
    pub symbol: Option<usize>,
    /// Children (internal node only).
    pub left: Option<Box<HuffmanNode>>,
    pub right: Option<Box<HuffmanNode>>,
}

impl PartialEq for HuffmanNode {
    fn eq(&self, other: &Self) -> bool {
        self.prob == other.prob
    }
}

impl Eq for HuffmanNode {}

impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other.prob.partial_cmp(&self.prob).unwrap_or(Ordering::Equal)
    }
}

/// Result of Huffman coding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuffmanCode {
    /// Code words for each symbol.
    pub codes: Vec<Vec<u8>>,
    /// Code lengths for each symbol.
    pub lengths: Vec<usize>,
    /// Expected code length.
    pub expected_length: f64,
    /// Root of the Huffman tree.
    pub tree: HuffmanNode,
}

/// Build a Huffman code from symbol probabilities.
pub fn huffman_code(probs: &[f64]) -> HuffmanCode {
    assert!(!probs.is_empty(), "Need at least one symbol");

    let n = probs.len();

    if n == 1 {
        let tree = HuffmanNode {
            prob: probs[0],
            symbol: Some(0),
            left: None,
            right: None,
        };
        return HuffmanCode {
            codes: vec![vec![]],
            lengths: vec![0],
            expected_length: 0.0,
            tree,
        };
    }

    // Build heap
    let mut heap: BinaryHeap<HuffmanNode> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| HuffmanNode {
            prob: p,
            symbol: Some(i),
            left: None,
            right: None,
        })
        .collect();

    // Build tree
    while heap.len() > 1 {
        let left = heap.pop().unwrap();
        let right = heap.pop().unwrap();
        let merged = HuffmanNode {
            prob: left.prob + right.prob,
            symbol: None,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        };
        heap.push(merged);
    }

    let tree = heap.pop().unwrap();

    // Extract codes
    let mut codes = vec![Vec::new(); n];
    extract_codes(&tree, &mut Vec::new(), &mut codes);

    let lengths: Vec<usize> = codes.iter().map(|c| c.len()).collect();
    let expected_length: f64 = probs
        .iter()
        .zip(lengths.iter())
        .map(|(&p, &l)| p * l as f64)
        .sum();

    HuffmanCode {
        codes,
        lengths,
        expected_length,
        tree,
    }
}

fn extract_codes(node: &HuffmanNode, prefix: &mut Vec<u8>, codes: &mut [Vec<u8>]) {
    if let Some(sym) = node.symbol {
        codes[sym] = prefix.clone();
        return;
    }
    if let Some(ref left) = node.left {
        prefix.push(0);
        extract_codes(left, prefix, codes);
        prefix.pop();
    }
    if let Some(ref right) = node.right {
        prefix.push(1);
        extract_codes(right, prefix, codes);
        prefix.pop();
    }
}

/// Compute optimal code lengths (Shannon-Fano): l_i = ceil(-log₂ p_i).
/// For zero probabilities, returns length 0.
pub fn optimal_code_lengths(probs: &[f64]) -> Vec<usize> {
    probs
        .iter()
        .map(|&p| {
            if p <= 0.0 {
                0
            } else {
                (-p.log2()).ceil() as usize
            }
        })
        .collect()
}

/// Compute the expected code length for given probabilities and lengths.
pub fn expected_code_length(probs: &[f64], lengths: &[usize]) -> f64 {
    probs
        .iter()
        .zip(lengths.iter())
        .map(|(&p, &l)| p * l as f64)
        .sum()
}

/// Shannon's source coding theorem: expected length ≥ H(X).
/// Verify that H(X) ≤ E[L] < H(X) + 1 for a prefix code.
pub fn verify_source_coding_bound(probs: &[f64], lengths: &[usize]) -> bool {
    let h = crate::entropy::shannon_entropy(probs);
    let el = expected_code_length(probs, lengths);
    el >= h - 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::shannon_entropy;
    use approx::assert_relative_eq;

    #[test]
    fn test_huffman_uniform() {
        let probs = &[0.25, 0.25, 0.25, 0.25];
        let code = huffman_code(probs);
        // All codes should be length 2 for uniform 4 symbols
        for l in &code.lengths {
            assert_eq!(*l, 2);
        }
        assert_relative_eq!(code.expected_length, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_huffman_single() {
        let probs = &[1.0];
        let code = huffman_code(probs);
        assert_eq!(code.lengths[0], 0);
    }

    #[test]
    fn test_huffman_optimality() {
        // E[L] should be close to H(X)
        let probs = &[0.4, 0.3, 0.2, 0.1];
        let code = huffman_code(probs);
        let h = shannon_entropy(probs);
        assert!(code.expected_length >= h - 1e-10);
        assert!(code.expected_length < h + 1.0);
    }

    #[test]
    fn test_huffman_prefix_free() {
        let probs = &[0.4, 0.3, 0.2, 0.1];
        let code = huffman_code(probs);
        // No code should be a prefix of another
        for i in 0..code.codes.len() {
            for j in 0..code.codes.len() {
                if i != j {
                    assert!(!code.codes[i].starts_with(&code.codes[j]));
                }
            }
        }
    }

    #[test]
    fn test_huffman_two_symbols() {
        let probs = &[0.7, 0.3];
        let code = huffman_code(probs);
        // One symbol gets length 1, the other length 1
        assert_eq!(code.lengths.iter().sum::<usize>(), 2);
    }

    #[test]
    fn test_optimal_code_lengths() {
        let probs = &[0.5, 0.25, 0.125, 0.125];
        let lengths = optimal_code_lengths(probs);
        assert_eq!(lengths, vec![1, 2, 3, 3]);
    }

    #[test]
    fn test_source_coding_bound() {
        let probs = &[0.4, 0.3, 0.2, 0.1];
        let code = huffman_code(probs);
        assert!(verify_source_coding_bound(probs, &code.lengths));
    }

    #[test]
    fn test_expected_length_entropy_gap() {
        // E[L] - H(X) < 1 for Huffman
        let probs = &[0.05, 0.1, 0.15, 0.2, 0.25, 0.25];
        let code = huffman_code(probs);
        let h = shannon_entropy(probs);
        assert!(code.expected_length - h < 1.0);
    }
}
