//! Information-theoretic inequalities: Fano's inequality, data processing inequality.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::entropy::{conditional_entropy, shannon_entropy};
use crate::mutual_info::mutual_information;

/// Fano's inequality: H(X|Y) ≤ H_b(Pe) + Pe * log₂(|X| - 1)
/// where Pe is the probability of error in estimating X from Y.
/// Returns the Fano bound.
pub fn fano_inequality(error_prob: f64, alphabet_size: usize) -> f64 {
    if alphabet_size <= 1 {
        return 0.0;
    }
    let h_b = binary_entropy_func(error_prob);
    h_b + error_prob * ((alphabet_size - 1) as f64).log2()
}

/// Binary entropy function H_b(p) = -p log₂ p - (1-p) log₂(1-p).
fn binary_entropy_func(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        0.0
    } else {
        shannon_entropy(&[p, 1.0 - p])
    }
}

/// Verify Fano's inequality for a given joint distribution.
/// Returns (H(X|Y), Fano bound, is_satisfied).
pub fn verify_fano_inequality(
    marginal_x: &[f64],
    joint_xy: &DMatrix<f64>,
) -> FanoResult {
    let h_x_given_y = conditional_entropy(marginal_x, joint_xy);

    // Compute minimum error probability: Pe = 1 - max_x P(x|y) marginalized
    let _n_x = joint_xy.nrows();
    let n_y = joint_xy.ncols();
    let mut pe = 1.0;
    for j in 0..n_y {
        let col_sum: f64 = joint_xy.column(j).iter().sum();
        if col_sum > 0.0 {
            let max_prob = joint_xy
                .column(j)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            pe -= col_sum * max_prob;
        }
    }

    let alphabet_size = marginal_x.len();
    let fano_bound = fano_inequality(pe, alphabet_size);

    FanoResult {
        conditional_entropy: h_x_given_y,
        error_probability: pe,
        fano_bound,
        is_satisfied: h_x_given_y <= fano_bound + 1e-10,
    }
}

/// Result of Fano's inequality verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanoResult {
    pub conditional_entropy: f64,
    pub error_probability: f64,
    pub fano_bound: f64,
    pub is_satisfied: bool,
}

/// Data processing inequality: if X → Y → Z forms a Markov chain, then I(X;Z) ≤ I(X;Y).
/// This function verifies the DPI given the three pairwise joint distributions.
pub fn verify_data_processing_inequality(
    marginal_x: &[f64],
    marginal_y: &[f64],
    marginal_z: &[f64],
    joint_xy: &DMatrix<f64>,
    joint_xz: &DMatrix<f64>,
) -> DataProcessingResult {
    let i_xy = mutual_information(marginal_x, marginal_y, joint_xy);
    let i_xz = mutual_information(marginal_x, marginal_z, joint_xz);

    DataProcessingResult {
        mutual_info_xy: i_xy,
        mutual_info_xz: i_xz,
        dpi_holds: i_xz <= i_xy + 1e-10,
    }
}

/// Result of data processing inequality verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessingResult {
    pub mutual_info_xy: f64,
    pub mutual_info_xz: f64,
    pub dpi_holds: bool,
}

/// Verify data processing inequality for a Markov chain X → Y → Z.
/// Takes the full joint distribution P(X,Y,Z) as a flattened matrix.
/// Dimensions: (|X|, |Y|*|Z|) or similar. For simplicity, takes pairwise joints.
pub fn data_processing_inequality_markov(
    px: &[f64],
    py: &[f64],
    pz: &[f64],
    pxy: &DMatrix<f64>,
    pxz: &DMatrix<f64>,
) -> bool {
    let i_xy = mutual_information(px, py, pxy);
    let i_xz = mutual_information(px, pz, pxz);
    i_xz <= i_xy + 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fano_bound_known() {
        // Pe=0.5, |X|=2: H_b(0.5) + 0.5 * log₂(1) = 1.0 + 0 = 1.0
        let bound = fano_inequality(0.5, 2);
        assert_relative_eq!(bound, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fano_zero_error() {
        // Pe=0: bound = 0
        let bound = fano_inequality(0.0, 4);
        assert_relative_eq!(bound, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fano_increasing_with_error() {
        let b1 = fano_inequality(0.1, 4);
        let b2 = fano_inequality(0.3, 4);
        let b3 = fano_inequality(0.5, 4);
        assert!(b1 <= b2);
        assert!(b2 <= b3);
    }

    #[test]
    fn test_fano_verified() {
        // Deterministic Y=X: Pe=0, H(X|Y)=0
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let joint = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let result = verify_fano_inequality(px, &joint);
        assert!(result.is_satisfied);
        assert_relative_eq!(result.conditional_entropy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fano_noisy_channel() {
        // BSC with p=0.1
        let px = &[0.5, 0.5];
        let joint = DMatrix::from_row_slice(2, 2, &[0.45, 0.05, 0.05, 0.45]);
        let result = verify_fano_inequality(px, &joint);
        assert!(result.is_satisfied);
    }

    #[test]
    fn test_dpi_identity_channel() {
        // Y=X: I(X;Y) = H(X), Z=Y=X: I(X;Z) = H(X)
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let pz = &[0.5, 0.5];
        let pxy = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let pxz = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let result = verify_data_processing_inequality(px, py, pz, &pxy, &pxz);
        assert!(result.dpi_holds);
    }

    #[test]
    fn test_dpi_noisy_processing() {
        // X → Y (clean), Y → Z (noisy): I(X;Z) ≤ I(X;Y)
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let pz = &[0.5, 0.5];
        // X-Y: identity (perfect correlation)
        let pxy = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        // X-Z: noisy (independent)
        let pxz = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
        let result = verify_data_processing_inequality(px, py, pz, &pxy, &pxz);
        assert!(result.dpi_holds);
        assert!(result.mutual_info_xz <= result.mutual_info_xy + 1e-10);
    }

    #[test]
    fn test_dpi_markov_chain() {
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let pz = &[0.5, 0.5];
        let pxy = DMatrix::from_row_slice(2, 2, &[0.4, 0.1, 0.1, 0.4]);
        let pxz = DMatrix::from_row_slice(2, 2, &[0.35, 0.15, 0.15, 0.35]);
        assert!(data_processing_inequality_markov(px, py, pz, &pxy, &pxz));
    }

    #[test]
    fn test_dpi_information_decrease() {
        // More processing = less information
        let px = &[0.5, 0.5];
        let py = &[0.5, 0.5];
        let pz = &[0.5, 0.5];
        let pxy = DMatrix::from_row_slice(2, 2, &[0.4, 0.1, 0.1, 0.4]);
        let pxz = DMatrix::from_row_slice(2, 2, &[0.3, 0.2, 0.2, 0.3]);
        let i_xy = mutual_information(px, py, &pxy);
        let i_xz = mutual_information(px, pz, &pxz);
        assert!(i_xz <= i_xy + 1e-10);
    }
}
