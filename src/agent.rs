//! Agent communication analysis: measuring information flow between agents.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::entropy::shannon_entropy;
use crate::mutual_info::mutual_information;

/// Represents an agent in a communication system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Agent identifier.
    pub id: String,
    /// Probability distribution over agent's states/messages.
    pub state_distribution: Vec<f64>,
}

impl Agent {
    /// Create a new agent.
    pub fn new(id: impl Into<String>, state_distribution: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            state_distribution,
        }
    }

    /// Compute the entropy of this agent's state.
    pub fn entropy(&self) -> f64 {
        shannon_entropy(&self.state_distribution)
    }
}

/// Represents a communication channel between two agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentChannel {
    /// Sender agent index.
    pub sender: usize,
    /// Receiver agent index.
    pub receiver: usize,
    /// Joint probability distribution of (sender, receiver) states.
    pub joint: DMatrix<f64>,
    /// Mutual information across this channel.
    pub mutual_information: f64,
    /// Channel efficiency: I(X;Y) / min(H(X), H(Y)).
    pub efficiency: f64,
}

/// Result of analyzing information flow in a multi-agent system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFlowAnalysis {
    /// Agents in the system.
    pub agents: Vec<Agent>,
    /// Communication channels between agents.
    pub channels: Vec<AgentChannel>,
    /// Total information in the system (sum of individual entropies).
    pub total_entropy: f64,
    /// Total mutual information across all channels.
    pub total_mutual_information: f64,
    /// Information flow efficiency.
    pub flow_efficiency: f64,
}

/// Analyze information flow between two agents.
pub fn analyze_agent_pair(
    sender: &Agent,
    receiver: &Agent,
    joint: &DMatrix<f64>,
) -> AgentChannel {
    let mi = mutual_information(&sender.state_distribution, &receiver.state_distribution, joint);
    let min_h = sender.entropy().min(receiver.entropy());
    let efficiency = if min_h > 0.0 { mi / min_h } else { 0.0 };

    AgentChannel {
        sender: 0, // placeholder
        receiver: 0,
        joint: joint.clone(),
        mutual_information: mi,
        efficiency,
    }
}

/// Compute the information bottleneck trade-off for agent communication.
/// For a given compression level β, find the optimal representation T that
/// balances compression I(T;X) and relevance I(T;Y).
pub fn information_bottleneck(
    px: &[f64],
    py_given_x: &DMatrix<f64>,
    n_clusters: usize,
    beta: f64,
    max_iterations: usize,
) -> InformationBottleneckResult {
    let n_x = px.len();
    let n_y = py_given_x.nrows();

    // Initialize cluster assignments randomly-ish (uniform)
    let mut pt_given_x = DMatrix::zeros(n_clusters, n_x);
    for x in 0..n_x {
        let cluster = x % n_clusters;
        pt_given_x[(cluster, x)] = 1.0;
    }

    for _ in 0..max_iterations {
        // Compute p(t)
        let mut pt = vec![0.0; n_clusters];
        for t in 0..n_clusters {
            for x in 0..n_x {
                pt[t] += pt_given_x[(t, x)] * px[x];
            }
        }

        // Compute p(y|t)
        let mut py_given_t = DMatrix::zeros(n_y, n_clusters);
        for t in 0..n_clusters {
            if pt[t] > 1e-15 {
                for y in 0..n_y {
                    for x in 0..n_x {
                        py_given_t[(y, t)] +=
                            pt_given_x[(t, x)] * px[x] * py_given_x[(y, x)] / pt[t];
                    }
                }
            }
        }

        // Update p(t|x) using IB equations
        let mut new_pt_given_x = DMatrix::zeros(n_clusters, n_x);
        for x in 0..n_x {
            let mut log_probs = vec![f64::NEG_INFINITY; n_clusters];
            for t in 0..n_clusters {
                if pt[t] > 1e-15 {
                    let mut kl = 0.0;
                    for y in 0..n_y {
                        let pyx = py_given_x[(y, x)];
                        let pyt: f64 = py_given_t[(y, t)];
                        if pyx > 0.0 && pyt > 0.0 {
                            let ratio: f64 = pyx / pyt;
                    kl += pyx * ratio.ln();
                        }
                    }
                    log_probs[t] = pt[t].ln() - beta * kl;
                }
            }

            // Log-sum-exp normalization
            let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_log > f64::NEG_INFINITY {
                let sum: f64 = log_probs
                    .iter()
                    .map(|&lp| (lp - max_log).exp())
                    .sum();
                for t in 0..n_clusters {
                    new_pt_given_x[(t, x)] = (log_probs[t] - max_log).exp() / sum;
                }
            }
        }

        pt_given_x = new_pt_given_x;
    }

    // Compute final metrics
    let mut pt = vec![0.0; n_clusters];
    for t in 0..n_clusters {
        for x in 0..n_x {
            pt[t] += pt_given_x[(t, x)] * px[x];
        }
    }

    let compression = {
        let mut c = 0.0;
        for t in 0..n_clusters {
            for x in 0..n_x {
                let ptx = pt_given_x[(t, x)];
                if ptx > 0.0 && pt[t] > 0.0 && px[x] > 0.0 {
                    c += ptx * px[x] * (ptx / pt[t]).log2();
                }
            }
        }
        c
    };

    InformationBottleneckResult {
        pt_given_x,
        pt,
        compression,
        beta,
    }
}

/// Result of information bottleneck computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationBottleneckResult {
    /// Cluster assignment probabilities P(T|X).
    pub pt_given_x: DMatrix<f64>,
    /// Cluster probabilities P(T).
    pub pt: Vec<f64>,
    /// Compression I(T;X).
    pub compression: f64,
    /// Trade-off parameter β.
    pub beta: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_agent_entropy() {
        let agent = Agent::new("A", vec![0.5, 0.5]);
        assert_relative_eq!(agent.entropy(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_agent_pair_noisy_channel() {
        let sender = Agent::new("sender", vec![0.5, 0.5]);
        let receiver = Agent::new("receiver", vec![0.5, 0.5]);
        let joint = DMatrix::from_row_slice(2, 2, &[0.45, 0.05, 0.05, 0.45]);
        let channel = analyze_agent_pair(&sender, &receiver, &joint);
        assert!(channel.mutual_information > 0.0);
        assert!(channel.mutual_information < 1.0);
        assert!(channel.efficiency > 0.0 && channel.efficiency <= 1.0);
    }

    #[test]
    fn test_agent_pair_perfect() {
        let sender = Agent::new("sender", vec![0.5, 0.5]);
        let receiver = Agent::new("receiver", vec![0.5, 0.5]);
        let joint = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let channel = analyze_agent_pair(&sender, &receiver, &joint);
        assert_relative_eq!(channel.mutual_information, 1.0, epsilon = 1e-10);
        assert_relative_eq!(channel.efficiency, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_agent_pair_independent() {
        let sender = Agent::new("sender", vec![0.5, 0.5]);
        let receiver = Agent::new("receiver", vec![0.5, 0.5]);
        let joint = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
        let channel = analyze_agent_pair(&sender, &receiver, &joint);
        assert_relative_eq!(channel.mutual_information, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_information_bottleneck() {
        let px = vec![0.5, 0.5];
        let py_given_x = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]);
        let result = information_bottleneck(&px, &py_given_x, 2, 1.0, 50);
        assert!(result.compression >= 0.0);
    }
}
