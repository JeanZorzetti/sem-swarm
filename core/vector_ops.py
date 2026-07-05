"""
SEM-Swarm — Vector Operations (Hybrid Architecture)
═══════════════════════════════════════════════════
Provides high-performance vector math for the epistemic memory.
Attempts to load the Rust PyO3 extension (`sem_vector`) for 10-15x
faster execution (crucial for batch deduplication and clustering).
Falls back to pure Python / numpy if the Rust extension is not compiled.

Usage:
    from core.vector_ops import batch_cosine, find_duplicates
"""

import logging
from typing import Any

logger = logging.getLogger("sem-swarm.vector")

# Try to load the Rust extension
try:
    import sem_vector
    RUST_ACCELERATED = True
    logger.info("⚡ Using Rust-accelerated sem_vector extension")
except ImportError:
    RUST_ACCELERATED = False
    logger.warning("⚠️ Rust extension 'sem_vector' not found. Falling back to pure Python.")

# ── Fallback Implementations ─────────────────────────────────

def _py_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure Python fallback for cosine similarity."""
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))

def _py_batch_cosine(query: list[float], targets: list[list[float]]) -> list[tuple[int, float]]:
    """Pure Python fallback for batch cosine."""
    results = []
    for i, target in enumerate(targets):
        results.append((i, _py_cosine_similarity(query, target)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def _py_find_duplicates(vectors: list[list[float]], threshold: float = 0.95) -> list[dict[str, Any]]:
    """Pure Python fallback for duplicate detection (O(N²))."""
    n = len(vectors)
    dups = []
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = _py_cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                dups.append({
                    "idx_a": i,
                    "idx_b": j,
                    "similarity": sim
                })
                
    dups.sort(key=lambda x: x["similarity"], reverse=True)
    return dups

def _py_find_contradictions(
    vectors: list[list[float]], 
    topic_threshold: float = 0.70, 
    duplicate_threshold: float = 0.95
) -> list[dict[str, Any]]:
    """Pure Python fallback for contradiction heuristics."""
    n = len(vectors)
    contras = []
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = _py_cosine_similarity(vectors[i], vectors[j])
            if sim >= topic_threshold and sim < duplicate_threshold:
                midpoint = (topic_threshold + duplicate_threshold) / 2.0
                dist = abs(sim - midpoint)
                rng = (duplicate_threshold - topic_threshold) / 2.0
                score = max(0.0, 1.0 - dist / rng) if rng > 0 else 0.5
                
                contras.append({
                    "idx_a": i,
                    "idx_b": j,
                    "embedding_similarity": sim,
                    "contradiction_score": score
                })
                
    contras.sort(key=lambda x: x["contradiction_score"], reverse=True)
    return contras

def _py_cluster_duplicates(vectors: list[list[float]], threshold: float = 0.95) -> list[list[int]]:
    """Pure Python fallback for union-find clustering."""
    n = len(vectors)
    parent = list(range(n))
    
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            
    dups = _py_find_duplicates(vectors, threshold)
    for dup in dups:
        union(dup["idx_a"], dup["idx_b"])
        
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
        
    return [g for g in groups.values() if len(g) > 1]

# ── Public API ───────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if RUST_ACCELERATED:
        return sem_vector.cosine_similarity(a, b)
    return _py_cosine_similarity(a, b)

def batch_cosine(query: list[float], targets: list[list[float]]) -> list[tuple[int, float]]:
    if RUST_ACCELERATED:
        return sem_vector.batch_cosine(query, targets)
    return _py_batch_cosine(query, targets)

def find_duplicates(vectors: list[list[float]], threshold: float = 0.95) -> list[dict[str, Any]]:
    if RUST_ACCELERATED:
        # Note: PyO3 objects have attributes but we return dicts in Python fallback
        # Let's map PyO3 objects to dicts for API compatibility
        dups = sem_vector.find_duplicates(vectors, threshold)
        return [{"idx_a": d.idx_a, "idx_b": d.idx_b, "similarity": d.similarity} for d in dups]
    return _py_find_duplicates(vectors, threshold)

def find_contradictions(
    vectors: list[list[float]], 
    topic_threshold: float = 0.70, 
    duplicate_threshold: float = 0.95
) -> list[dict[str, Any]]:
    if RUST_ACCELERATED:
        contras = sem_vector.find_contradictions(vectors, topic_threshold, duplicate_threshold)
        return [
            {
                "idx_a": c.idx_a, 
                "idx_b": c.idx_b, 
                "embedding_similarity": c.embedding_similarity, 
                "contradiction_score": c.contradiction_score
            } 
            for c in contras
        ]
    return _py_find_contradictions(vectors, topic_threshold, duplicate_threshold)

def cluster_duplicates(vectors: list[list[float]], threshold: float = 0.95) -> list[list[int]]:
    if RUST_ACCELERATED:
        return sem_vector.cluster_duplicates(vectors, threshold)
    return _py_cluster_duplicates(vectors, threshold)
