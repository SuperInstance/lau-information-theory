//! Mutual information and normalized variants.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::entropy::{joint_entropy, shannon_entropy};

/// Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
/// Takes marginals and joint distribution.
pub fn mutual_information(marginal_x: &[f64], marginal_y: &[f64], joint: &DMatrix<f64>) -> f64 {
    let hx = shannon_entropy(marginal_x);
    let hy = shannon_entropy(marginal_y);
    let hxy = joint_entropy(joint);
    hx + hy - hxy
}

/// Mutual information via KL divergence: I(X;Y) = D_KL(P_XY || P_X × P_Y).
pub fn mutual_information_kl(marginal_x: &[f64], marginal_y: &[f64], joint: &DMatrix<f64>) -> f64 {
    let mut mi = 0.0;
    for i in 0..joint.nrows() {
        for j in 0..joint.ncols() {
            let pxy = joint[(i, j)];
            if pxy > 0.0 {
                let px = marginal_x.get(i).copied().unwrap_or(0.0);
                let py = marginal_y.get(j).copied().unwrap_or(0.0);
                if px > 0.0 && py > 0.0 {
                    mi += pxy * (pxy / (px * py)).log2();
                }
            }
        }
    }
    mi
}

/// Normalized mutual information: I(X;Y) / H(X).
pub fn normalized_mutual_information_x(
    marginal_x: &[f64],
    marginal_y: &[f64],
    joint: &DMatrix<f64>,
) -> f64 {
    let hx = shannon_entropy(marginal_x);
    if hx == 0.0 {
        return 0.0;
    }
    mutual_information(marginal_x, marginal_y, joint) / hx
}

/// Normalized mutual information: I(X;Y) / sqrt(H(X) * H(Y)).
pub fn normalized_mutual_information_sqrt(
    marginal_x: &[f64],
    marginal_y: &[f64],
    joint: &DMatrix<f64>,
) -> f64 {
    let hx = shannon_entropy(marginal_x);
    let hy = shannon_entropy(marginal_y);
    let denom = (hx * hy).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    mutual_information(marginal_x, marginal_y, joint) / denom
}

/// Variation of information: VI(X,Y) = H(X|Y) + H(Y|X).
pub fn variation_of_information(
    marginal_x: &[f64],
    marginal_y: &[f64],
    joint: &DMatrix<f64>,
) -> f64 {
    let hxy = joint_entropy(joint);
    let hx = shannon_entropy(marginal_x);
    let hy = shannon_entropy(marginal_y);
    2.0 * hxy - hx - hy
}

/// Conditional mutual information I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z).
/// For simplicity, takes the 3-way joint as flattened and dimension sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualInfoResult {
    pub mi: f64,
    pub nmi: f64,
    pub vi: f64,
}

/// Compute all mutual information measures at once.
pub fn analyze_mutual_information(
    marginal_x: &[f64],
    marginal_y: &[f64],
    joint: &DMatrix<f64>,
) -> MutualInfoResult {
    let mi = mutual_information(marginal_x, marginal_y, joint);
    let hx = shannon_entropy(marginal_x);
    let hy = shannon_entropy(marginal_y);
    let nmi = if (hx * hy).sqrt() > 0.0 {
        mi / (hx * hy).sqrt()
    } else {
        0.0
    };
    let vi = variation_of_information(marginal_x, marginal_y, joint);
    MutualInfoResult { mi, nmi, vi }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mutual_information_independent() {
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let joint = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
        assert_relative_eq!(
            mutual_information(px, py, &joint),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mutual_information_perfect_correlation() {
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let joint = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let mi = mutual_information(px, py, &joint);
        assert_relative_eq!(mi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_nonnegative() {
        let px = &[0.6, 0.4];
        let py = &[0.3, 0.7];
        let joint = DMatrix::from_row_slice(2, 2, &[0.2, 0.4, 0.1, 0.3]);
        assert!(mutual_information(px, py, &joint) >= 0.0);
    }

    #[test]
    fn test_mutual_information_leq_entropy() {
        // I(X;Y) ≤ min(H(X), H(Y))
        let px = &[0.6, 0.4];
        let py = &[0.3, 0.7];
        let joint = DMatrix::from_row_slice(2, 2, &[0.2, 0.4, 0.1, 0.3]);
        let mi = mutual_information(px, py, &joint);
        let hx = shannon_entropy(px);
        let hy = shannon_entropy(py);
        assert!(mi <= hx + 1e-10);
        assert!(mi <= hy + 1e-10);
    }

    #[test]
    fn test_mutual_information_kl_matches() {
        let px = &[0.6, 0.4];
        let py = &[0.3, 0.7];
        let joint = DMatrix::from_row_slice(2, 2, &[0.2, 0.4, 0.1, 0.3]);
        let mi1 = mutual_information(px, py, &joint);
        let mi2 = mutual_information_kl(px, py, &joint);
        assert_relative_eq!(mi1, mi2, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_range() {
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let joint = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let nmi = normalized_mutual_information_sqrt(px, py, &joint);
        assert_relative_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variation_of_information_independent() {
        // For independent X,Y: VI = H(X) + H(Y) - 2*I(X;Y) = H(X) + H(Y)
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let joint = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
        let vi = variation_of_information(px, py, &joint);
        assert_relative_eq!(vi, 2.0, epsilon = 1e-10); // H(X)+H(Y) = 2
    }
}
