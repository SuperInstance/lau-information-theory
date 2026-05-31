//! Channel capacity: discrete memoryless channels, BSC, BEC.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::entropy::shannon_entropy;

/// A discrete memoryless channel defined by a transition matrix.
/// channel_matrix[y][x] = P(Y=y | X=x), i.e., columns sum to 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteMemorylessChannel {
    /// Transition probability matrix. Rows = output symbols, cols = input symbols.
    pub transition: DMatrix<f64>,
}

impl DiscreteMemorylessChannel {
    /// Create a new DMC. Validates column-stochastic property.
    pub fn new(transition: DMatrix<f64>) -> Result<Self, String> {
        // Check non-negative
        if transition.iter().any(|&v| v < 0.0) {
            return Err("Transition probabilities must be non-negative".into());
        }
        // Check columns sum to ~1
        for j in 0..transition.ncols() {
            let col_sum: f64 = transition.column(j).iter().sum();
            if (col_sum - 1.0).abs() > 1e-9 {
                return Err(format!("Column {} sums to {}, expected 1.0", j, col_sum));
            }
        }
        Ok(Self { transition })
    }

    /// Compute mutual information I(X;Y) for input distribution p_x.
    pub fn mutual_info_for_input(&self, p_x: &[f64]) -> f64 {
        let n_outputs = self.transition.nrows();
        let n_inputs = self.transition.ncols();

        // Compute output distribution
        let mut p_y = vec![0.0; n_outputs];
        for y in 0..n_outputs {
            for x in 0..n_inputs {
                p_y[y] += self.transition[(y, x)] * p_x[x];
            }
        }

        // I(X;Y) = H(Y) - H(Y|X)
        let h_y = shannon_entropy(&p_y);

        // H(Y|X) = Σ p(x) H(Y|X=x)
        let mut h_y_given_x = 0.0;
        for x in 0..n_inputs {
            if p_x[x] > 0.0 {
                let cond_probs: Vec<f64> =
                    (0..n_outputs).map(|y| self.transition[(y, x)]).collect();
                h_y_given_x += p_x[x] * shannon_entropy(&cond_probs);
            }
        }

        h_y - h_y_given_x
    }

    /// Compute channel capacity using the Blahut-Arimoto algorithm.
    /// Returns (capacity, optimal input distribution).
    pub fn capacity(&self, max_iterations: usize, tolerance: f64) -> (f64, Vec<f64>) {
        let n = self.transition.ncols();
        let mut q = vec![1.0 / n as f64; n]; // uniform initial

        for _ in 0..max_iterations {
            // Compute output distribution
            let mut p_y = vec![0.0; self.transition.nrows()];
            for y in 0..self.transition.nrows() {
                for x in 0..n {
                    p_y[y] += self.transition[(y, x)] * q[x];
                }
            }

            // Compute c(x) = exp(Σ_y P(y|x) log(P(y|x)/p(y)))
            let mut c = vec![0.0; n];
            for x in 0..n {
                for y in 0..self.transition.nrows() {
                    let pyx = self.transition[(y, x)];
                    if pyx > 0.0 && p_y[y] > 0.0 {
                        c[x] += pyx * (pyx / p_y[y]).log2();
                    }
                }
                c[x] = 2.0_f64.powf(c[x]);
            }

            // Update q
            let c_sum: f64 = c.iter().zip(q.iter()).map(|(&ci, &qi)| ci * qi).sum();
            let q_new: Vec<f64> = c
                .iter()
                .zip(q.iter())
                .map(|(&ci, &qi)| ci * qi / c_sum)
                .collect();

            // Check convergence
            let delta: f64 = q_new
                .iter()
                .zip(q.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            q = q_new;
            if delta < tolerance {
                break;
            }
        }

        let capacity = self.mutual_info_for_input(&q);
        (capacity, q)
    }
}

/// Binary symmetric channel with crossover probability p.
pub fn bsc_channel(crossover_prob: f64) -> DiscreteMemorylessChannel {
    let q = 1.0 - crossover_prob;
    let transition = DMatrix::from_row_slice(
        2,
        2,
        &[q, crossover_prob, crossover_prob, q],
    );
    DiscreteMemorylessChannel { transition }
}

/// BSC capacity: C = 1 - H(p) where H is binary entropy.
pub fn bsc_capacity(crossover_prob: f64) -> f64 {
    1.0 - binary_entropy(crossover_prob)
}

/// Binary entropy H(p) = -p log₂ p - (1-p) log₂(1-p).
pub fn binary_entropy(p: f64) -> f64 {
    shannon_entropy(&[p, 1.0 - p])
}

/// Binary erasure channel with erasure probability ε.
pub fn bec_channel(erasure_prob: f64) -> DiscreteMemorylessChannel {
    // Input: {0, 1}, Output: {0, 1, e}
    // P(0|0) = 1-ε, P(e|0) = ε, P(1|1) = 1-ε, P(e|1) = ε
    let q = 1.0 - erasure_prob;
    let e = erasure_prob;
    let transition = DMatrix::from_row_slice(
        3,
        2,
        &[
            q, 0.0, // output 0
            0.0, q, // output 1
            e, e,   // output e (erasure)
        ],
    );
    DiscreteMemorylessChannel { transition }
}

/// BEC capacity: C = 1 - ε.
pub fn bec_capacity(erasure_prob: f64) -> f64 {
    1.0 - erasure_prob
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bsc_capacity_noiseless() {
        assert_relative_eq!(bsc_capacity(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bsc_capacity_half() {
        assert_relative_eq!(bsc_capacity(0.5), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bsc_capacity_known() {
        // BSC with p=0.1: C = 1 - H(0.1)
        let expected = 1.0 - binary_entropy(0.1);
        assert_relative_eq!(bsc_capacity(0.1), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_bsc_capacity_range() {
        // Capacity should be in [0, 1] for crossover in [0, 0.5]
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
            let c = bsc_capacity(p);
            assert!(c >= -1e-10 && c <= 1.0 + 1e-10, "p={}: c={}", p, c);
        }
    }

    #[test]
    fn test_bec_capacity() {
        assert_relative_eq!(bec_capacity(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(bec_capacity(0.5), 0.5, epsilon = 1e-10);
        assert_relative_eq!(bec_capacity(1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dmc_capacity_bsc() {
        let ch = bsc_channel(0.1);
        let (cap, _) = ch.capacity(1000, 1e-12);
        let expected = bsc_capacity(0.1);
        assert_relative_eq!(cap, expected, epsilon = 0.01);
    }

    #[test]
    fn test_dmc_capacity_bec() {
        let ch = bec_channel(0.3);
        let (cap, _) = ch.capacity(1000, 1e-12);
        let expected = bec_capacity(0.3);
        assert_relative_eq!(cap, expected, epsilon = 0.01);
    }

    #[test]
    fn test_binary_entropy_known() {
        // H(0.5) = 1
        assert_relative_eq!(binary_entropy(0.5), 1.0, epsilon = 1e-10);
        // H(0) = 0, H(1) = 0
        assert_relative_eq!(binary_entropy(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(binary_entropy(1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dmc_construction_valid() {
        let t = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]);
        assert!(DiscreteMemorylessChannel::new(t).is_ok());
    }

    #[test]
    fn test_dmc_construction_invalid() {
        // Columns sum to 2.0, not 1.0
        let t = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        assert!(DiscreteMemorylessChannel::new(t).is_err());
    }
}
