//! # lau-information-theory
//!
//! Shannon information theory: entropy, mutual information, channel capacity,
//! source coding, and agent communication analysis.

pub mod entropy;
pub mod divergence;
pub mod mutual_info;
pub mod channel;
pub mod coding;
pub mod kraft;
pub mod rate_distortion;
pub mod inequalities;
pub mod agent;

pub use entropy::*;
pub use divergence::*;
pub use mutual_info::*;
pub use channel::*;
pub use coding::*;
pub use kraft::*;
pub use rate_distortion::*;
pub use inequalities::*;
pub use agent::*;
