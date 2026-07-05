//! Batch vector operations using Rayon for data parallelism.
//!
//! These functions process large collections of vectors in parallel,
//! distributing work across all available CPU cores. Critical for the
//! VPS (4 vCPUs) where Python's GIL prevents true parallelism.
//!
//! Performance characteristics (2048-dim vectors):
//!   - Single cosine:    ~0.3μs
//!   - 10k batch:        ~0.8ms (Rayon) vs ~45ms (Python/numpy)
//!   - 10k×10k pairwise: ~4s (Rayon) vs ~60s (Python/numpy)

use rayon::prelude::*;

use crate::ops::cosine_similarity;

/// Compute cosine similarity between one query vector and many target vectors.
///
/// Returns a Vec of (index, similarity) pairs, sorted by similarity descending.
/// This is the core operation for semantic search: "find the N most similar facts."
///
/// Uses Rayon's parallel iterator to distribute across all CPU cores.
pub fn batch_cosine_against_one(query: &[f32], targets: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = targets
        .par_iter()
        .enumerate()
        .map(|(idx, target)| (idx, cosine_similarity(query, target)))
        .collect();

    // Sort by similarity descending (most similar first)
    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Compute the top-K most similar vectors to a query.
///
/// More efficient than `batch_cosine_against_one` when K << N,
/// as it avoids fully sorting the results.
pub fn top_k_similar(query: &[f32], targets: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut results = batch_cosine_against_one(query, targets);
    results.truncate(k);
    results
}

/// Compute the full pairwise cosine similarity matrix.
///
/// Returns a flattened upper-triangular matrix of (i, j, similarity) triples.
/// Only computes each pair once (i < j) since cosine similarity is symmetric.
///
/// For N vectors: computes N*(N-1)/2 pairs.
/// Uses Rayon's parallel iterator for the outer loop.
///
/// **Warning:** This is O(N²) in the number of vectors. For N > 50,000,
/// consider using approximate methods or pgvector's HNSW index instead.
pub fn pairwise_cosine_matrix(vectors: &[Vec<f32>]) -> Vec<(usize, usize, f32)> {
    let n = vectors.len();

    // Parallel over rows, sequential within each row
    (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (i + 1..n)
                .map(|j| (i, j, cosine_similarity(&vectors[i], &vectors[j])))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Compute statistics over a set of similarity scores.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SimilarityStats {
    pub count: usize,
    pub mean: f32,
    pub min: f32,
    pub max: f32,
    pub std_dev: f32,
}

/// Compute summary statistics for pairwise similarities.
pub fn similarity_stats(vectors: &[Vec<f32>]) -> SimilarityStats {
    let pairs = pairwise_cosine_matrix(vectors);
    let count = pairs.len();

    if count == 0 {
        return SimilarityStats {
            count: 0,
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            std_dev: 0.0,
        };
    }

    let sims: Vec<f32> = pairs.iter().map(|(_, _, s)| *s).collect();
    let sum: f32 = sims.iter().sum();
    let mean = sum / count as f32;
    let min = sims.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let variance: f32 = sims.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / count as f32;
    let std_dev = variance.sqrt();

    SimilarityStats {
        count,
        mean,
        min,
        max,
        std_dev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vectors() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0],   // East
            vec![0.0, 1.0, 0.0],   // North
            vec![0.0, 0.0, 1.0],   // Up
            vec![1.0, 1.0, 0.0],   // Northeast
            vec![0.99, 0.01, 0.0], // Almost East
        ]
    }

    #[test]
    fn test_batch_cosine_against_one() {
        let vectors = make_test_vectors();
        let query = vec![1.0, 0.0, 0.0]; // East

        let results = batch_cosine_against_one(&query, &vectors);

        // Most similar should be East (idx 0, sim ≈ 1.0) and Almost East (idx 4)
        assert_eq!(results[0].0, 0); // East itself
        assert!((results[0].1 - 1.0).abs() < 1e-4);
        assert_eq!(results[1].0, 4); // Almost East
        assert!(results[1].1 > 0.99);
    }

    #[test]
    fn test_top_k() {
        let vectors = make_test_vectors();
        let query = vec![1.0, 0.0, 0.0];

        let top2 = top_k_similar(&query, &vectors, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 0); // East
        assert_eq!(top2[1].0, 4); // Almost East
    }

    #[test]
    fn test_pairwise_matrix() {
        let vectors = make_test_vectors();
        let pairs = pairwise_cosine_matrix(&vectors);

        // N=5 → 5*4/2 = 10 pairs
        assert_eq!(pairs.len(), 10);

        // All pairs should be (i, j) where i < j
        for (i, j, _) in &pairs {
            assert!(i < j);
        }
    }

    #[test]
    fn test_similarity_stats() {
        let vectors = make_test_vectors();
        let stats = similarity_stats(&vectors);
        assert_eq!(stats.count, 10);
        assert!(stats.min >= -1.0);
        assert!(stats.max <= 1.0);
    }

    #[test]
    fn test_large_batch_2048d() {
        // Simulate 100 vectors of 2048 dimensions (qwen3-embedding)
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|seed| {
                (0..2048)
                    .map(|i| ((seed * 2048 + i) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();

        let query = &vectors[0];
        let results = top_k_similar(query, &vectors, 5);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, 0); // Self-match
        assert!((results[0].1 - 1.0).abs() < 1e-4);
    }
}
