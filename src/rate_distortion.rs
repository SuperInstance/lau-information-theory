//! Rate-distortion theory basics.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::entropy::shannon_entropy;

/// A distortion matrix D where D[i][j] is the distortion when symbol i is reproduced as j.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistortionMatrix {
    pub values: DMatrix<f64>,
}

/// Compute the rate-distortion function R(D) using the Blahut algorithm.
/// This is a simplified implementation for a Bernoulli source with Hamming distortion.
pub fn rate_distortion_bernoulli(source_prob: f64, max_distortion: f64) -> f64 {
    // For a Bernoulli(p) source with Hamming distortion:
    // R(D) = H(p) - H(D) for 0 ≤ D ≤ min(p, 1-p)
    // R(D) = 0 for D ≥ min(p, 1-p)
    let hp = binary_entropy_safe(source_prob);
    let d_max = source_prob.min(1.0 - source_prob);
    if max_distortion >= d_max {
        return 0.0;
    }
    if max_distortion <= 0.0 {
        return hp;
    }
    hp - binary_entropy_safe(max_distortion)
}

fn binary_entropy_safe(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        0.0
    } else {
        shannon_entropy(&[p, 1.0 - p])
    }
}

/// Compute the distortion-rate function D(R) for a Gaussian source
/// with mean-squared error distortion.
/// D(R) = σ² * 2^(-2R)
pub fn distortion_rate_gaussian(variance: f64, rate: f64) -> f64 {
    variance * 2.0_f64.powf(-2.0 * rate)
}

/// Rate-distortion function for a Gaussian source with MSE distortion.
/// R(D) = 0.5 * log₂(σ²/D) for D < σ², 0 otherwise.
pub fn rate_distortion_gaussian(variance: f64, distortion: f64) -> f64 {
    if distortion >= variance {
        return 0.0;
    }
    if distortion <= 0.0 {
        return f64::INFINITY;
    }
    0.5 * (variance / distortion).log2()
}

/// Result of rate-distortion analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateDistortionResult {
    pub rate: f64,
    pub distortion: f64,
    pub source_entropy: f64,
}

/// Compute the rate-distortion curve at multiple distortion levels.
pub fn rate_distortion_curve_bernoulli(
    source_prob: f64,
    distortion_levels: &[f64],
) -> Vec<RateDistortionResult> {
    let h_source = binary_entropy_safe(source_prob);
    distortion_levels
        .iter()
        .map(|&d| RateDistortionResult {
            rate: rate_distortion_bernoulli(source_prob, d),
            distortion: d,
            source_entropy: h_source,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rate_distortion_bernoulli_zero_distortion() {
        // R(0) = H(p)
        let p = 0.5;
        let r = rate_distortion_bernoulli(p, 0.0);
        assert_relative_eq!(r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rate_distortion_bernoulli_max_distortion() {
        // R(D_max) = 0
        let p = 0.3;
        let r = rate_distortion_bernoulli(p, 0.3);
        assert_relative_eq!(r, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rate_distortion_bernoulli_monotone() {
        let p = 0.4;
        let r0 = rate_distortion_bernoulli(p, 0.05);
        let r1 = rate_distortion_bernoulli(p, 0.15);
        let r2 = rate_distortion_bernoulli(p, 0.35);
        assert!(r0 >= r1);
        assert!(r1 >= r2);
    }

    #[test]
    fn test_distortion_rate_gaussian() {
        let variance = 1.0;
        let d = distortion_rate_gaussian(variance, 1.0);
        assert_relative_eq!(d, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_rate_distortion_gaussian() {
        let r = rate_distortion_gaussian(1.0, 0.25);
        assert_relative_eq!(r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rate_distortion_gaussian_zero() {
        let r = rate_distortion_gaussian(1.0, 2.0);
        assert_relative_eq!(r, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_curve_bernoulli() {
        let curve = rate_distortion_curve_bernoulli(0.5, &[0.0, 0.1, 0.25, 0.5]);
        assert_eq!(curve.len(), 4);
        assert!(curve[0].rate >= curve[1].rate);
    }
}
