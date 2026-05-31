//! Kraft inequality verification.

use serde::{Deserialize, Serialize};

/// Verify the Kraft inequality: Σ 2^(-l_i) ≤ 1 for a prefix-free code.
pub fn verify_kraft_inequality(lengths: &[usize]) -> KraftResult {
    let sum: f64 = lengths
        .iter()
        .filter(|&&l| l > 0)
        .map(|&l| 2.0_f64.powi(-(l as i32)))
        .sum();

    KraftResult {
        lengths: lengths.to_vec(),
        kraft_sum: sum,
        is_valid: sum <= 1.0 + 1e-10,
    }
}

/// Compute the Kraft sum for given code lengths.
pub fn kraft_sum(lengths: &[usize]) -> f64 {
    lengths
        .iter()
        .filter(|&&l| l > 0)
        .map(|&l| 2.0_f64.powi(-(l as i32)))
        .sum()
}

/// Find optimal code lengths that satisfy Kraft inequality with equality.
/// Uses the Shannon-Fano approach: l_i = ceil(-log₂ p_i) and adjusts.
pub fn kraft_optimal_lengths(probs: &[f64]) -> Vec<usize> {
    let mut lengths: Vec<usize> = probs
        .iter()
        .map(|&p| {
            if p <= 0.0 {
                0
            } else {
                (-p.log2()).ceil() as usize
            }
        })
        .collect();

    // Verify Kraft holds; if sum > 1, increase longest codes
    while kraft_sum(&lengths) > 1.0 + 1e-10 {
        // Find the longest code and increase it
        if let Some(idx) = lengths.iter().enumerate().max_by_key(|&(_, &l)| l).map(|(i, _)| i) {
            lengths[idx] += 1;
        }
    }

    lengths
}

/// Result of Kraft inequality verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KraftResult {
    /// Code lengths.
    pub lengths: Vec<usize>,
    /// Kraft sum Σ 2^(-l_i).
    pub kraft_sum: f64,
    /// Whether the Kraft inequality is satisfied.
    pub is_valid: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kraft_valid_code() {
        // Lengths [1, 2, 3, 3]: sum = 0.5 + 0.25 + 0.125 + 0.125 = 1.0
        let result = verify_kraft_inequality(&[1, 2, 3, 3]);
        assert!(result.is_valid);
        assert_relative_eq!(result.kraft_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kraft_invalid_code() {
        // Lengths [1, 1, 1]: sum = 0.5 + 0.5 + 0.5 = 1.5 > 1
        let result = verify_kraft_inequality(&[1, 1, 1]);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_kraft_trivial() {
        // Single code of length 1
        let result = verify_kraft_inequality(&[1]);
        assert!(result.is_valid);
    }

    #[test]
    fn test_kraft_empty() {
        let result = verify_kraft_inequality(&[]);
        assert!(result.is_valid);
        assert_relative_eq!(result.kraft_sum, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kraft_huffman_satisfies() {
        use crate::coding::huffman_code;
        let probs = &[0.4, 0.3, 0.2, 0.1];
        let code = huffman_code(probs);
        let result = verify_kraft_inequality(&code.lengths);
        assert!(result.is_valid);
    }

    #[test]
    fn test_kraft_optimal_lengths() {
        let probs = &[0.5, 0.25, 0.125, 0.125];
        let lengths = kraft_optimal_lengths(probs);
        let result = verify_kraft_inequality(&lengths);
        assert!(result.is_valid);
    }

    #[test]
    fn test_kraft_sum_known() {
        assert_relative_eq!(kraft_sum(&[1, 2, 3, 3]), 1.0, epsilon = 1e-10);
    }
}
