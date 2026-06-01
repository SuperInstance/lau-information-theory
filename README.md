# lau-information-theory

> Shannon information theory: entropy, mutual information, channel capacity, source coding, and agent communication analysis

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

## What This Does

A complete Shannon information theory library that implements the core mathematical toolkit:

- **Entropy**: Discrete Shannon entropy, differential entropy (Gaussian, uniform), joint entropy, conditional entropy
- **Divergences**: KL divergence, Jensen-Shannon divergence, cross-entropy, divergence matrices
- **Mutual information**: MI via entropy, MI via KL divergence, normalized variants, variation of information
- **Channel capacity**: Discrete memoryless channels, Blahut-Arimoto algorithm, BSC, BEC
- **Source coding**: Huffman coding, optimal code lengths, Kraft inequality verification
- **Rate-distortion theory**: Blahut-Arimoto rate-distortion, distortion-rate function
- **Information inequalities**: Fano's inequality, data processing inequality verification
- **Agent communication**: Agent-to-agent channel analysis, information flow tracing

All functions work with probability distributions as `&[f64]` slices and joint distributions as `nalgebra::DMatrix<f64>`. Every type is serializable via serde.

## Key Idea

Shannon's 1948 paper established that information is quantifiable. This crate makes those quantifiers composable: you can compute entropy, then mutual information from it, then channel capacity from that, and verify the theoretical inequalities that bind them together at each step.

The library enforces mathematical consistency: Fano's inequality bounds the conditional entropy given an error probability, the data processing inequality shows that post-processing can only lose information, and the Kraft inequality guarantees prefix-free codes.

## Install

```toml
[dependencies]
lau-information-theory = "0.1"
```

### Dependencies

- **nalgebra** 0.33 (with `serde-serialize`) — matrices for joint distributions
- **serde** 1 (with `derive`) — serialization

Dev dependency: **approx** 0.5 for test assertions.

## Quick Start

### Entropy

```rust
use lau_information_theory::*;

// Shannon entropy H(X) in bits
let h = shannon_entropy(&[0.5, 0.5]); // 1.0 bit (fair coin)

// Joint entropy from a joint probability matrix
let joint = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
let h_xy = joint_entropy(&joint); // 2.0 bits (independent fair coins)

// Conditional entropy H(Y|X) = H(X,Y) - H(X)
let px = &[0.5, 0.5];
let h_y_given_x = conditional_entropy(px, &joint);

// Differential entropy for Gaussians
let h_gauss = differential_entropy_gaussian(1.0); // h(N(0,1)) = ½ log₂(2πe)

// Custom log base (e.g., nats instead of bits)
let h_nats = shannon_entropy_with_base(&[0.5, 0.5], std::f64::consts::E);
```

### Mutual Information

```rust
let px = &[0.5, 0.5];
let py = &[0.5, 0.5];

// Perfect correlation: I(X;Y) = 1 bit
let joint = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
let mi = mutual_information(px, py, &joint); // 1.0

// Independent: I(X;Y) = 0
let independent = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
let mi0 = mutual_information(px, py, &independent); // 0.0

// Full analysis: MI + normalized MI + variation of information
let result = analyze_mutual_information(px, py, &joint);
println!("MI: {}, NMI: {}, VI: {}", result.mi, result.nmi, result.vi);
```

### Channel Capacity (Blahut-Arimoto)

```rust
// Define a binary symmetric channel with flip probability 0.1
let bsc = DiscreteMemorylessChannel::new(
    DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9])
).unwrap();

// Compute capacity via Blahut-Arimoto
let (capacity, optimal_input) = bsc.capacity(1000, 1e-10);
println!("BSC(0.1) capacity: {:.4} bits", capacity);
```

### Huffman Coding

```rust
let probs = &[0.4, 0.3, 0.2, 0.1];
let code = huffman_code(probs);

println!("Expected length: {:.2} bits", code.expected_length);
println!("Entropy:         {:.2} bits", shannon_entropy(probs));
// Source coding theorem: H(X) ≤ E[L] < H(X) + 1
assert!(verify_source_coding_bound(probs, &code.lengths));
```

### Kraft Inequality

```rust
// Verify that code lengths satisfy Kraft inequality: Σ 2^(-lᵢ) ≤ 1
let lengths = &[1, 2, 3, 3];
assert!(verify_kraft_inequality(lengths));
```

## API Reference

### Entropy (`entropy` module)

| Function | Signature | Description |
|----------|-----------|-------------|
| `shannon_entropy` | `(&[f64]) → f64` | H(X) = −Σ p(x) log₂ p(x) |
| `shannon_entropy_with_base` | `(&[f64], base: f64) → f64` | Entropy with custom log base |
| `differential_entropy_gaussian` | `(variance: f64) → f64` | h(N(μ, σ²)) = ½ log₂(2πeσ²) |
| `differential_entropy_uniform` | `(a: f64, b: f64) → f64` | h(U(a,b)) = log₂(b−a) |
| `joint_entropy` | `(&DMatrix<f64>) → f64` | H(X,Y) from joint distribution |
| `conditional_entropy` | `(&[f64], &DMatrix<f64>) → f64` | H(Y\|X) = H(X,Y) − H(X) |

### Divergences (`divergence` module)

| Function | Signature | Description |
|----------|-----------|-------------|
| `kl_divergence` | `(&[f64], &[f64]) → f64` | D_KL(P\|Q) = Σ p log(p/q) |
| `js_divergence` | `(&[f64], &[f64]) → f64` | Symmetric JSD = ½ D_KL(P\|M) + ½ D_KL(Q\|M) |
| `cross_entropy` | `(&[f64], &[f64]) → f64` | H(P,Q) = −Σ p log q |
| `DivergenceMatrix` | struct | KL divergence for all pairs of distributions |

### Mutual Information (`mutual_info` module)

| Function | Signature | Description |
|----------|-----------|-------------|
| `mutual_information` | `(marg_x, marg_y, joint) → f64` | I(X;Y) = H(X) + H(Y) − H(X,Y) |
| `mutual_information_kl` | `(marg_x, marg_y, joint) → f64` | I(X;Y) via KL divergence |
| `normalized_mutual_information_sqrt` | `(…) → f64` | I(X;Y) / √(H(X)·H(Y)) |
| `variation_of_information` | `(…) → f64` | VI(X,Y) = 2H(X,Y) − H(X) − H(Y) |
| `analyze_mutual_information` | `(…) → MutualInfoResult` | MI + NMI + VI in one call |

### Channel Capacity (`channel` module)

| Type/Function | Description |
|---------------|-------------|
| `DiscreteMemorylessChannel` | Channel defined by a column-stochastic transition matrix |
| `.capacity(max_iter, tol)` | Blahut-Arimoto: returns (capacity, optimal input distribution) |
| `.mutual_info_for_input(&[f64])` | I(X;Y) for a given input distribution |

### Source Coding (`coding` module)

| Function | Description |
|----------|-------------|
| `huffman_code(&[f64])` | Build optimal Huffman code → `HuffmanCode` |
| `optimal_code_lengths(&[f64])` | Shannon-Fano lengths: ⌈−log₂ pᵢ⌉ |
| `expected_code_length(probs, lengths)` | E[L] = Σ pᵢ lᵢ |
| `verify_source_coding_bound(probs, lengths)` | H(X) ≤ E[L] |

### Kraft Inequality (`kraft` module)

| Function | Description |
|----------|-------------|
| `verify_kraft_inequality(&[usize])` | Σ 2^(−lᵢ) ≤ 1 |
| `kraft_sum(&[usize])` | Compute Σ 2^(−lᵢ) |
| `kraft_mcmillan_theorem(probs, lengths)` | Verify complete prefix code conditions |

### Rate-Distortion (`rate_distortion` module)

| Function | Description |
|----------|-------------|
| `blahut_arimoto_rate_distortion(…)` | Compute R(D) via Blahut-Arimoto |
| `rate_distortion_function(probs, distortion, max_iter, tol)` | Theoretical R(D) curve |

### Inequalities (`inequalities` module)

| Function | Description |
|----------|-------------|
| `fano_inequality(Pe, \|X\|)` | H(X\|Y) ≤ H_b(Pe) + Pe log₂(\|X\|−1) |
| `verify_fano_inequality(px, joint)` | Verify Fano bound on actual data |
| `verify_data_processing_inequality(…)` | I(X;Z) ≤ I(X;Y) for Markov chain X→Y→Z |

## How It Works

1. **Entropy module**: Direct summation of −p log₂ p with zero-probability filtering. Joint entropy sums over all entries of the joint distribution matrix. Conditional entropy uses the chain rule H(Y|X) = H(X,Y) − H(X).

2. **Channel capacity**: The Blahut-Arimoto algorithm iterates between computing the output distribution p(y) = Σ_x P(y|x) q(x) and updating the input distribution q(x) ∝ exp(Σ_y P(y|x) log(P(y|x)/p(y))). Converges to the capacity-achieving input distribution.

3. **Huffman coding**: Builds a binary tree using a min-heap. At each step, the two lowest-probability nodes merge. The resulting tree gives optimal prefix-free codes satisfying H(X) ≤ E[L] < H(X) + 1.

4. **Fano's inequality**: Given an estimator with error probability Pe over |X| symbols, bounds the conditional entropy: H(X|Y) ≤ H_b(Pe) + Pe · log₂(|X|−1), where H_b is the binary entropy function.

## The Math

### Shannon Entropy

```
H(X) = −Σₓ p(x) log₂ p(x)  [bits]
```

Properties: H(X) ≥ 0, H(X) ≤ log₂ |X| (uniform maximizes), H(X,X) = H(X).

### Mutual Information

```
I(X;Y) = H(X) + H(Y) − H(X,Y) = D_KL(P_XY ‖ P_X × P_Y)
```

Range: 0 ≤ I(X;Y) ≤ min(H(X), H(Y)). Zero iff X ⊥ Y.

### Channel Capacity

```
C = max_{p(x)} I(X;Y)
```

For a BSC with flip probability p: C = 1 − H_b(p).

### Source Coding Theorem

For any prefix-free code with lengths l₁, …, lₙ:

```
H(X) ≤ E[L] < H(X) + 1    (Shannon's source coding theorem)
Σ 2^(−lᵢ) ≤ 1              (Kraft inequality, necessary and sufficient)
```

### Fano's Inequality

```
H(X|Y) ≤ H_b(Pe) + Pe · log₂(|X| − 1)
```

Where Pe = P(Ĥ ≠ H) is the probability of estimation error and H_b(p) = −p log₂ p − (1−p) log₂(1−p).

### Data Processing Inequality

For a Markov chain X → Y → Z:

```
I(X;Z) ≤ I(X;Y)
```

No processing of Y can increase information about X.

## Tests

72 unit tests covering:
- Entropy of uniform, deterministic, and empty distributions
- Differential entropy for Gaussian and uniform distributions
- Joint/conditional entropy chain rules
- KL divergence: non-negativity, asymmetry, known values
- Jensen-Shannon symmetry and boundedness
- Mutual information: independence (0), perfect correlation (= H(X)), KL equivalence
- Huffman coding: optimality, prefix-free property, source coding bounds
- Kraft inequality verification
- Channel capacity for BSC and BEC
- Fano's inequality verification with noisy channels
- Data processing inequality for Markov chains

Run with: `cargo test`

## License

MIT
