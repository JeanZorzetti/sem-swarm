"""
Test script to verify Python <-> Rust hybrid integration.
"""
import time
import random
from core.vector_ops import (
    RUST_ACCELERATED, 
    cosine_similarity, 
    batch_cosine,
    find_duplicates,
    cluster_duplicates
)

def test_hybrid_vector_ops():
    print(f"Rust Acceleration Active: {RUST_ACCELERATED}")
    
    # 1. Test basic cosine similarity
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    print(f"Cosine [1,0,0] vs [0,1,0]: {cosine_similarity(v1, v2)}")
    
    # 2. Test batch operations with "large" synthetic data
    print("\nGenerating 5000 random 2048-dimensional vectors...")
    start_gen = time.time()
    
    # Create 5000 random 2048-dim vectors
    # We make a few deliberate duplicates to test the dedup
    dim = 2048
    n_vectors = 5000
    
    vectors = []
    base_vector = [random.random() for _ in range(dim)]
    
    for i in range(n_vectors):
        if i in (10, 20, 30):
            # Near-duplicates of base_vector
            vectors.append([v + (random.random() * 0.001) for v in base_vector])
        else:
            vectors.append([random.random() for _ in range(dim)])
            
    print(f"Generation took {time.time() - start_gen:.2f}s")
    
    # Test batch cosine
    print("\nTesting batch_cosine (1 against 5000)...")
    start_batch = time.time()
    results = batch_cosine(base_vector, vectors)
    print(f"Batch cosine took {time.time() - start_batch:.4f}s")
    
    print("Top 5 matches:")
    for idx, sim in results[:5]:
        print(f"  Index {idx}: sim={sim:.4f}")
        
    # Test dedup
    print("\nTesting find_duplicates (N=5000, threshold=0.95)...")
    print("This requires N*(N-1)/2 = 12.5M comparisons.")
    start_dedup = time.time()
    dups = find_duplicates(vectors, 0.95)
    print(f"Dedup took {time.time() - start_dedup:.4f}s")
    print(f"Found {len(dups)} duplicate pairs.")
    for d in dups[:5]:
        print(f"  {d}")
        
    # Test clustering
    print("\nTesting cluster_duplicates...")
    clusters = cluster_duplicates(vectors, 0.95)
    print(f"Found {len(clusters)} clusters:")
    for c in clusters:
        print(f"  {c}")

if __name__ == "__main__":
    test_hybrid_vector_ops()
