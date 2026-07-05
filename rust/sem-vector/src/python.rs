//! PyO3 bindings — exposes Rust vector operations to Python.
//!
//! Usage from Python:
//! ```python
//! from sem_vector import (
//!     cosine_similarity,
//!     batch_cosine,
//!     top_k_similar,
//!     find_duplicates,
//!     find_contradictions,
//!     cluster_duplicates,
//! )
//!
//! # Single similarity
//! sim = cosine_similarity([1.0, 0.0], [0.0, 1.0])  # 0.0
//!
//! # Batch: find 5 most similar vectors to a query
//! results = top_k_similar(query_vec, all_vectors, k=5)
//!
//! # Find duplicate facts (similarity > 0.95)
//! dups = find_duplicates(embeddings, threshold=0.95)
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::{batch, dedup, ops};

/// Register all Python-visible functions in the `sem_vector` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(py_top_k_similar, m)?)?;
    m.add_function(wrap_pyfunction!(py_find_duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(py_find_contradictions, m)?)?;
    m.add_function(wrap_pyfunction!(py_cluster_duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(py_pairwise_matrix, m)?)?;
    Ok(())
}

// ── Single Operations ────────────────────────────────────────

/// Compute cosine similarity between two vectors.
///
/// Args:
///     a: First vector (list of floats)
///     b: Second vector (list of floats)
///
/// Returns:
///     Cosine similarity value in [-1.0, 1.0]
#[pyfunction]
#[pyo3(name = "cosine_similarity")]
fn py_cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "Vector dimensions must match: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(ops::cosine_similarity(&a, &b))
}

// ── Batch Operations ─────────────────────────────────────────

/// Compute cosine similarity between a query vector and many targets.
///
/// Returns list of (index, similarity) tuples sorted by similarity descending.
///
/// Args:
///     query: Query vector
///     targets: List of target vectors
///
/// Returns:
///     List of (index, similarity) tuples
#[pyfunction]
#[pyo3(name = "batch_cosine")]
fn py_batch_cosine(query: Vec<f32>, targets: Vec<Vec<f32>>) -> PyResult<Vec<(usize, f32)>> {
    validate_dimensions(&query, &targets)?;
    Ok(batch::batch_cosine_against_one(&query, &targets))
}

/// Find the top-K most similar vectors to a query.
///
/// Args:
///     query: Query vector
///     targets: List of target vectors
///     k: Number of results to return
///
/// Returns:
///     List of (index, similarity) tuples, length <= k
#[pyfunction]
#[pyo3(name = "top_k_similar")]
fn py_top_k_similar(query: Vec<f32>, targets: Vec<Vec<f32>>, k: usize) -> PyResult<Vec<(usize, f32)>> {
    validate_dimensions(&query, &targets)?;
    Ok(batch::top_k_similar(&query, &targets, k))
}

/// Compute the full pairwise cosine similarity matrix.
///
/// Returns list of (i, j, similarity) triples for all pairs where i < j.
///
/// WARNING: O(N²) complexity. Use only for N < 50,000.
#[pyfunction]
#[pyo3(name = "pairwise_matrix")]
fn py_pairwise_matrix(vectors: Vec<Vec<f32>>) -> PyResult<Vec<(usize, usize, f32)>> {
    Ok(batch::pairwise_cosine_matrix(&vectors))
}

// ── Dedup Operations ─────────────────────────────────────────

/// Find duplicate fact pairs based on embedding similarity.
///
/// Args:
///     vectors: List of embedding vectors
///     threshold: Minimum cosine similarity to consider as duplicate (default: 0.95)
///
/// Returns:
///     List of dicts with keys: idx_a, idx_b, similarity
#[pyfunction]
#[pyo3(name = "find_duplicates", signature = (vectors, threshold=0.95))]
fn py_find_duplicates(
    vectors: Vec<Vec<f32>>,
    threshold: f32,
) -> PyResult<Vec<PyDuplicatePair>> {
    let dups = dedup::find_duplicates(&vectors, threshold);
    Ok(dups
        .into_iter()
        .map(|d| PyDuplicatePair {
            idx_a: d.idx_a,
            idx_b: d.idx_b,
            similarity: d.similarity,
        })
        .collect())
}

/// Find potentially contradictory fact pairs.
///
/// Uses a heuristic: facts with high topical similarity but below
/// the duplicate threshold are candidates for contradiction.
///
/// Args:
///     vectors: List of embedding vectors
///     topic_threshold: Minimum similarity to consider same-topic (default: 0.70)
///     duplicate_threshold: Above this, it's a duplicate not contradiction (default: 0.95)
///
/// Returns:
///     List of dicts with keys: idx_a, idx_b, embedding_similarity, contradiction_score
#[pyfunction]
#[pyo3(name = "find_contradictions", signature = (vectors, topic_threshold=0.70, duplicate_threshold=0.95))]
fn py_find_contradictions(
    vectors: Vec<Vec<f32>>,
    topic_threshold: f32,
    duplicate_threshold: f32,
) -> PyResult<Vec<PyContradictionPair>> {
    let contras = dedup::find_contradictions(&vectors, topic_threshold, duplicate_threshold);
    Ok(contras
        .into_iter()
        .map(|c| PyContradictionPair {
            idx_a: c.idx_a,
            idx_b: c.idx_b,
            embedding_similarity: c.embedding_similarity,
            contradiction_score: c.contradiction_score,
        })
        .collect())
}

/// Cluster duplicate facts using union-find.
///
/// Returns groups of indices that should be consolidated into single facts.
/// If A≈B and B≈C, then {A,B,C} form a cluster.
///
/// Args:
///     vectors: List of embedding vectors
///     threshold: Minimum cosine similarity (default: 0.95)
///
/// Returns:
///     List of lists of indices (each inner list is a cluster)
#[pyfunction]
#[pyo3(name = "cluster_duplicates", signature = (vectors, threshold=0.95))]
fn py_cluster_duplicates(vectors: Vec<Vec<f32>>, threshold: f32) -> PyResult<Vec<Vec<usize>>> {
    Ok(dedup::cluster_duplicates(&vectors, threshold))
}

// ── Python-visible structs ───────────────────────────────────

#[pyclass]
#[derive(Clone)]
struct PyDuplicatePair {
    #[pyo3(get)]
    idx_a: usize,
    #[pyo3(get)]
    idx_b: usize,
    #[pyo3(get)]
    similarity: f32,
}

#[pymethods]
impl PyDuplicatePair {
    fn __repr__(&self) -> String {
        format!(
            "DuplicatePair(idx_a={}, idx_b={}, similarity={:.4})",
            self.idx_a, self.idx_b, self.similarity
        )
    }
}

#[pyclass]
#[derive(Clone)]
struct PyContradictionPair {
    #[pyo3(get)]
    idx_a: usize,
    #[pyo3(get)]
    idx_b: usize,
    #[pyo3(get)]
    embedding_similarity: f32,
    #[pyo3(get)]
    contradiction_score: f32,
}

#[pymethods]
impl PyContradictionPair {
    fn __repr__(&self) -> String {
        format!(
            "ContradictionPair(idx_a={}, idx_b={}, sim={:.4}, score={:.4})",
            self.idx_a, self.idx_b, self.embedding_similarity, self.contradiction_score
        )
    }
}

// ── Helpers ──────────────────────────────────────────────────

fn validate_dimensions(query: &[f32], targets: &[Vec<f32>]) -> PyResult<()> {
    let dim = query.len();
    for (i, target) in targets.iter().enumerate() {
        if target.len() != dim {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: query has {} dims, target[{}] has {} dims",
                dim,
                i,
                target.len()
            )));
        }
    }
    Ok(())
}
