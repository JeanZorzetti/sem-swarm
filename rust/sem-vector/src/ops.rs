//! Core vector math operations.
//!
//! These are the fundamental building blocks: dot product, L2 norm,
//! and cosine similarity. Written to be auto-vectorizable by LLVM
//! into SIMD instructions (SSE4.2 / AVX2 on Intel i7).

/// Compute the dot product of two vectors.
///
/// # Panics
/// Panics if vectors have different lengths.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    // Written as a simple fold — LLVM auto-vectorizes this into SIMD
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the L2 (Euclidean) norm of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    dot_product(v, v).sqrt()
}

/// Compute the cosine similarity between two vectors.
///
/// Returns a value in [-1.0, 1.0] where:
/// - 1.0 = identical direction
/// - 0.0 = orthogonal (unrelated)
/// - -1.0 = opposite direction
///
/// Returns 0.0 if either vector has zero norm (avoids NaN).
///
/// # Panics
/// Panics if vectors have different lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    // Clamp to [-1.0, 1.0] to handle floating point imprecision
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute cosine distance (1.0 - cosine_similarity).
///
/// Returns a value in [0.0, 2.0] where:
/// - 0.0 = identical
/// - 1.0 = orthogonal
/// - 2.0 = opposite
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_high_dimensional() {
        // Simulate 2048-dim vectors (qwen3-embedding dimension)
        let a: Vec<f32> = (0..2048).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..2048).map(|i| (i as f32).cos()).collect();
        let sim = cosine_similarity(&a, &b);
        // Just verify it's in valid range and doesn't panic
        assert!((-1.0..=1.0).contains(&sim));
    }
}
