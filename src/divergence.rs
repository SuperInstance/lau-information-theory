//! KL divergence and related measures.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Compute KL divergence D_KL(P || Q) = Σ p(x) log₂(p(x)/q(x)).
/// Returns infinity if P assigns mass where Q has zero mass.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "Distributions must have same length");
    let mut div = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            if qi <= 0.0 {
                return f64::INFINITY;
            }
            div += pi * (pi / qi).log2();
        }
    }
    div
}

/// Compute Jensen-Shannon divergence JSD(P, Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
/// where M = 0.5*(P+Q). Always finite and symmetric.
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| 0.5 * (pi + qi)).collect();
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// Cross-entropy H(P, Q) = -Σ p(x) log₂ q(x).
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut h = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            if qi <= 0.0 {
                return f64::INFINITY;
            }
            h -= pi * qi.log2();
        }
    }
    h
}

/// Relative entropy matrix for all pairs of distributions (rows of the matrix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceMatrix {
    /// D[i][j] = D_KL(P_i || P_j)
    pub values: DMatrix<f64>,
}

impl DivergenceMatrix {
    /// Compute KL divergence for all pairs of distributions.
    pub fn from_distributions(distributions: &[Vec<f64>]) -> Self {
        let n = distributions.len();
        let mut values = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                values[(i, j)] = kl_divergence(&distributions[i], &distributions[j]);
            }
        }
        Self { values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![0.5, 0.5];
        assert_relative_eq!(kl_divergence(&p, &p), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_nonnegative() {
        let p = vec![0.6, 0.4];
        let q = vec![0.5, 0.5];
        assert!(kl_divergence(&p, &q) >= 0.0);
    }

    #[test]
    fn test_kl_divergence_asymmetric() {
        let p = vec![0.6, 0.4];
        let q = vec![0.5, 0.5];
        let kl_pq = kl_divergence(&p, &q);
        let kl_qp = kl_divergence(&q, &p);
        // KL is asymmetric: D(P||Q) ≠ D(Q||P) in general
        assert!((kl_pq - kl_qp).abs() > 1e-10);
    }

    #[test]
    fn test_kl_divergence_known_value() {
        let p = vec![0.5, 0.5];
        let q = vec![0.75, 0.25];
        // D_KL = 0.5*log(0.5/0.75) + 0.5*log(0.5/0.25)
        let expected = 0.5 * (0.5_f64 / 0.75).log2() + 0.5 * (0.5_f64 / 0.25).log2();
        assert_relative_eq!(kl_divergence(&p, &q), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_infinite() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        assert_eq!(kl_divergence(&p, &q), f64::INFINITY);
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = vec![0.6, 0.4];
        let q = vec![0.5, 0.5];
        assert_relative_eq!(
            js_divergence(&p, &q),
            js_divergence(&q, &p),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_js_divergence_identical() {
        let p = vec![0.3, 0.7];
        assert_relative_eq!(js_divergence(&p, &p), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cross_entropy_decomposition() {
        // H(P,Q) = H(P) + D_KL(P||Q)
        let p = vec![0.5, 0.5];
        let q = vec![0.75, 0.25];
        use crate::entropy::shannon_entropy;
        let expected = shannon_entropy(&p) + kl_divergence(&p, &q);
        assert_relative_eq!(cross_entropy(&p, &q), expected, epsilon = 1e-10);
    }
}
