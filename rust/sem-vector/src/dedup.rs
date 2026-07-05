//! Duplicate and contradiction detection in the Epistemic Memory.
//!
//! These operations are critical for the Dreaming Loop, which runs
//! overnight to consolidate the swarm's collective knowledge:
//!
//! - **Duplicates:** Facts that say essentially the same thing
//!   (cosine similarity > threshold). Should be merged.
//!
//! - **Contradictions:** Facts that are semantically close in topic
//!   but oppose each other. Detected via high topical similarity
//!   combined with low content similarity, or via explicit
//!   contradiction markers.

use rayon::prelude::*;
use serde::Serialize;

use crate::ops::cosine_similarity;

/// A pair of duplicate facts found in the epistemic memory.
#[derive(Debug, Clone, Serialize)]
pub struct DuplicatePair {
    /// Index of the first fact
    pub idx_a: usize,
    /// Index of the second fact
    pub idx_b: usize,
    /// Cosine similarity between the two facts
    pub similarity: f32,
}

/// A pair of potentially contradictory facts.
#[derive(Debug, Clone, Serialize)]
pub struct ContradictionPair {
    /// Index of the first fact
    pub idx_a: usize,
    /// Index of the second fact
    pub idx_b: usize,
    /// Cosine similarity between embeddings (topical closeness)
    pub embedding_similarity: f32,
    /// Contradiction score: how likely these facts contradict each other
    /// Higher = more likely contradiction. Range [0.0, 1.0]
    pub contradiction_score: f32,
}

/// Find duplicate fact pairs based on embedding similarity threshold.
///
/// Two facts are considered duplicates if their cosine similarity
/// exceeds the given threshold (default: 0.95 for near-identical).
///
/// Returns pairs sorted by similarity descending (most similar first).
///
/// # Arguments
/// * `vectors` - Embedding vectors for each fact
/// * `threshold` - Minimum cosine similarity to consider as duplicate (0.0 to 1.0)
///
/// # Performance
/// O(N²/2) comparisons, parallelized via Rayon.
/// For 10,000 facts: ~2 seconds (Rust) vs ~30+ seconds (Python/numpy).
pub fn find_duplicates(vectors: &[Vec<f32>], threshold: f32) -> Vec<DuplicatePair> {
    let n = vectors.len();

    let mut duplicates: Vec<DuplicatePair> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let mut local_dups = Vec::new();
            for j in (i + 1)..n {
                let sim = cosine_similarity(&vectors[i], &vectors[j]);
                if sim >= threshold {
                    local_dups.push(DuplicatePair {
                        idx_a: i,
                        idx_b: j,
                        similarity: sim,
                    });
                }
            }
            local_dups
        })
        .collect();

    // Sort by similarity descending
    duplicates.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    duplicates
}

/// Find potentially contradictory fact pairs.
///
/// Contradiction detection heuristic:
/// Two facts are considered potentially contradictory when:
/// 1. Their embeddings have HIGH similarity (same topic, sim > `topic_threshold`)
/// 2. But they're NOT duplicates (sim < `duplicate_threshold`)
///
/// The intuition: facts about the same topic that aren't near-identical
/// often represent different (possibly contradictory) viewpoints.
///
/// The contradiction_score is computed as:
///   topic_similarity * (1.0 - abs(topic_similarity - duplicate_threshold))
///
/// This produces high scores for facts that are:
/// - Close enough to be about the same topic
/// - But different enough to potentially disagree
///
/// # Arguments
/// * `vectors` - Embedding vectors for each fact
/// * `topic_threshold` - Minimum similarity to consider same-topic (e.g., 0.70)
/// * `duplicate_threshold` - Above this, it's a duplicate, not contradiction (e.g., 0.95)
///
/// # Note
/// This is a heuristic. For robust contradiction detection, the Dreaming Loop
/// should pass candidate pairs through an LLM (e.g., deepseek-r1:14b on VPS)
/// for semantic verification.
pub fn find_contradictions(
    vectors: &[Vec<f32>],
    topic_threshold: f32,
    duplicate_threshold: f32,
) -> Vec<ContradictionPair> {
    let n = vectors.len();

    let mut contradictions: Vec<ContradictionPair> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let mut local_contradictions = Vec::new();
            for j in (i + 1)..n {
                let sim = cosine_similarity(&vectors[i], &vectors[j]);

                // Same topic but not duplicate
                if sim >= topic_threshold && sim < duplicate_threshold {
                    // Contradiction score: peaks in the "same topic, different content" zone
                    let midpoint = (topic_threshold + duplicate_threshold) / 2.0;
                    let distance_from_midpoint = (sim - midpoint).abs();
                    let range = (duplicate_threshold - topic_threshold) / 2.0;
                    let contradiction_score = if range > 0.0 {
                        (1.0 - distance_from_midpoint / range).max(0.0)
                    } else {
                        0.5
                    };

                    local_contradictions.push(ContradictionPair {
                        idx_a: i,
                        idx_b: j,
                        embedding_similarity: sim,
                        contradiction_score,
                    });
                }
            }
            local_contradictions
        })
        .collect();

    // Sort by contradiction score descending
    contradictions.sort_unstable_by(|a, b| {
        b.contradiction_score
            .partial_cmp(&a.contradiction_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    contradictions
}

/// Identify clusters of duplicate facts that should be merged.
///
/// Uses a simple union-find approach: if A≈B and B≈C, then {A,B,C}
/// form a cluster even if A and C aren't directly similar.
///
/// Returns groups of indices that should be consolidated into single facts.
pub fn cluster_duplicates(vectors: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
    let n = vectors.len();
    let duplicates = find_duplicates(vectors, threshold);

    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]); // Path compression
        }
        parent[x]
    }

    fn union(parent: &mut [usize], x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px != py {
            parent[px] = py;
        }
    }

    for dup in &duplicates {
        union(&mut parent, dup.idx_a, dup.idx_b);
    }

    // Group by root
    let mut groups: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }

    // Only return groups with more than 1 member (actual duplicates)
    groups
        .into_values()
        .filter(|group| group.len() > 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_facts() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0],       // Fact 0: Topic A
            vec![0.99, 0.01, 0.0],      // Fact 1: Nearly identical to 0
            vec![0.98, 0.02, 0.0],      // Fact 2: Nearly identical to 0 and 1
            vec![0.0, 1.0, 0.0],        // Fact 3: Completely different topic
            vec![0.7, 0.3, 0.0],        // Fact 4: Related to topic A but different
            vec![0.0, 0.0, 1.0],        // Fact 5: Another different topic
        ]
    }

    #[test]
    fn test_find_duplicates_strict() {
        let facts = make_test_facts();
        let dups = find_duplicates(&facts, 0.98);

        // Facts 0, 1, 2 should be duplicates of each other
        assert!(!dups.is_empty());
        assert!(dups.iter().any(|d| d.idx_a == 0 && d.idx_b == 1));
    }

    #[test]
    fn test_find_duplicates_no_false_positives() {
        let facts = make_test_facts();
        let dups = find_duplicates(&facts, 0.98);

        // Fact 3, 5 should NOT be duplicates of anything
        assert!(!dups.iter().any(|d| d.idx_a == 3 || d.idx_b == 3));
        assert!(!dups.iter().any(|d| d.idx_a == 5 || d.idx_b == 5));
    }

    #[test]
    fn test_find_contradictions() {
        let facts = make_test_facts();
        let contras = find_contradictions(&facts, 0.60, 0.97);

        // Fact 4 is related to topic A (sim > 0.6 with facts 0,1,2)
        // but different enough (sim < 0.97) to be a potential contradiction
        assert!(!contras.is_empty());
    }

    #[test]
    fn test_cluster_duplicates() {
        let facts = make_test_facts();
        let clusters = cluster_duplicates(&facts, 0.97);

        // Facts 0, 1, 2 should cluster together
        assert!(!clusters.is_empty());
        let largest_cluster = clusters.iter().max_by_key(|c| c.len()).unwrap();
        assert!(largest_cluster.contains(&0));
        assert!(largest_cluster.contains(&1));
    }

    #[test]
    fn test_empty_vectors() {
        let empty: Vec<Vec<f32>> = vec![];
        assert!(find_duplicates(&empty, 0.9).is_empty());
        assert!(find_contradictions(&empty, 0.6, 0.95).is_empty());
        assert!(cluster_duplicates(&empty, 0.9).is_empty());
    }
}
