//! Entropy: discrete Shannon entropy, differential entropy, joint and conditional entropy.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Compute Shannon entropy H(X) = -Σ p(x) log₂ p(x) for a discrete distribution.
/// Returns 0 for empty or zero-probability distributions.
pub fn shannon_entropy(probs: &[f64]) -> f64 {
    if probs.is_empty() {
        return 0.0;
    }
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    -probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| {
            let normalized = p / total;
            normalized * normalized.log2()
        })
        .sum::<f64>()
}

/// Compute differential entropy h(X) = ∫ f(x) log₂ f(x) dx for a Gaussian distribution
/// with the given variance.
pub fn differential_entropy_gaussian(variance: f64) -> f64 {
    if variance <= 0.0 {
        return f64::NEG_INFINITY;
    }
    0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * variance).log2()
}

/// Compute uniform differential entropy h(X) = log₂(b - a) over [a, b].
pub fn differential_entropy_uniform(a: f64, b: f64) -> f64 {
    if b <= a {
        return f64::NEG_INFINITY;
    }
    (b - a).log2()
}

/// Joint entropy H(X,Y) from a joint probability matrix.
/// The matrix is organized with rows = X values, cols = Y values.
pub fn joint_entropy(joint: &DMatrix<f64>) -> f64 {
    let total: f64 = joint.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    -joint
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| {
            let n = p / total;
            n * n.log2()
        })
        .sum::<f64>()
}

/// Conditional entropy H(Y|X) = H(X,Y) - H(X).
/// Takes marginal of X and the joint distribution.
pub fn conditional_entropy(marginal_x: &[f64], joint: &DMatrix<f64>) -> f64 {
    joint_entropy(joint) - shannon_entropy(marginal_x)
}

/// Configuration for entropy computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyConfig {
    /// Base of the logarithm (default: 2 for bits).
    pub base: f64,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self { base: 2.0 }
    }
}

/// Compute Shannon entropy with a custom log base.
pub fn shannon_entropy_with_base(probs: &[f64], base: f64) -> f64 {
    if probs.is_empty() {
        return 0.0;
    }
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let log_base = base.log2();
    -probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| {
            let n = p / total;
            n * n.log2() / log_base
        })
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_entropy() {
        // Fair coin: H = 1 bit
        let probs = &[0.5, 0.5];
        assert_relative_eq!(shannon_entropy(probs), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fair_die_entropy() {
        // Fair 6-sided die: H = log₂(6)
        let probs = &[1.0 / 6.0; 6];
        assert_relative_eq!(shannon_entropy(probs), 6f64.log2(), epsilon = 1e-10);
    }

    #[test]
    fn test_deterministic_entropy() {
        // Deterministic: H = 0
        let probs = &[1.0, 0.0, 0.0];
        assert_relative_eq!(shannon_entropy(probs), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_entropy() {
        assert_relative_eq!(shannon_entropy(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_upper_bound() {
        // H(X) ≤ log₂(|X|) — uniform maximizes
        let n = 8;
        let uniform = vec![1.0 / n as f64; n];
        let skewed = vec![0.9, 0.1];
        assert!(shannon_entropy(&uniform) >= shannon_entropy(&skewed));
    }

    #[test]
    fn test_entropy_nonnegative() {
        let probs = &[0.3, 0.4, 0.3];
        assert!(shannon_entropy(probs) >= 0.0);
    }

    #[test]
    fn test_differential_entropy_gaussian_unit_variance() {
        // h(N(0,1)) = 0.5 * log₂(2πe)
        let expected = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).log2();
        assert_relative_eq!(
            differential_entropy_gaussian(1.0),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_differential_entropy_uniform() {
        // U(0,1): h = log₂(1) = 0
        assert_relative_eq!(differential_entropy_uniform(0.0, 1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_joint_entropy_independent() {
        // H(X,Y) = H(X) + H(Y) when independent
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let joint = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
        let h_x = shannon_entropy(px);
        let h_y = shannon_entropy(py);
        let h_xy = joint_entropy(&joint);
        assert_relative_eq!(h_xy, h_x + h_y, epsilon = 1e-10);
    }

    #[test]
    fn test_conditional_entropy() {
        // For independent X,Y: H(Y|X) = H(Y)
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let joint = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
        let h_y_given_x = conditional_entropy(px, &joint);
        let h_y = shannon_entropy(py);
        assert_relative_eq!(h_y_given_x, h_y, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_with_nats() {
        let probs = &[0.5, 0.5];
        let h_bits = shannon_entropy(probs);
        let h_nats = shannon_entropy_with_base(probs, std::f64::consts::E);
        assert_relative_eq!(h_nats, h_bits * 2.0_f64.ln(), epsilon = 1e-10);
    }
}
