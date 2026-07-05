//! # SEM-Swarm Vector Operations
//!
//! High-performance vector operations for the Shared Epistemic Memory.
//! This module provides SIMD-friendly cosine similarity, batch operations,
//! duplicate detection, and contradiction finding.
//!
//! All operations use Rayon for data parallelism across available CPU cores,
//! critical for the VPS (4 vCPUs) where Python's GIL prevents true parallelism.

/// Core vector math operations (cosine similarity, dot product, norms).
pub mod ops;
/// Batch operations over collections of vectors (parallel via Rayon).
pub mod batch;
/// Duplicate and contradiction detection in epistemic memory.
pub mod dedup;
/// PyO3 bindings for Python interop.
mod python;

pub use ops::{cosine_similarity, dot_product, l2_norm};
pub use batch::{batch_cosine_against_one, pairwise_cosine_matrix};
pub use dedup::{find_duplicates, find_contradictions, DuplicatePair, ContradictionPair};

use pyo3::prelude::*;

/// Python module: `sem_vector`
///
/// Usage from Python:
/// ```python
/// from sem_vector import cosine_similarity, batch_cosine, find_duplicates
/// ```
#[pymodule]
fn sem_vector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)?;
    Ok(())
}
