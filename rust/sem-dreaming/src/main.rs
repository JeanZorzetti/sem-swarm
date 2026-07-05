//! SEM-Swarm Dreaming Loop (Rust Implementation)
//!
//! Overnight consolidation of epistemic memory.
//! This process runs on the VPS (via cron or nohup) and performs:
//! 1. Memory deduplication (merging similar facts)
//! 2. Contradiction detection
//! 3. Re-indexing
//!
//! Uses `sem-vector` for high-performance SIMD math and Rayon
//! for parallelism across the 4 vCPUs.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use sqlx::{FromRow, Pool, Postgres};
use std::env;
use std::time::Instant;
use tracing::{error, info, Level};

use sem_vector::{cluster_duplicates, find_contradictions};

/// Database representation of a fact in Epistemic Memory
#[derive(Debug, FromRow)]
struct EpistemicFact {
    id: i64,
    fact_text: String,
    // pgvector returns vectors as generic text or specialized types.
    // For simplicity in this skeleton, we assume it's fetched as Vec<f32>.
    // In sqlx with pgvector, it's typically mapped to pgvector::Vector.
    #[sqlx(try_from = "pgvector::Vector")]
    embedding: pgvector::Vector,
    confidence_score: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🌙 Starting SEM-Swarm Dreaming Loop...");
    let start_time = Instant::now();

    // 1. Connect to Database
    let db_url = env::var("DATABASE_URL_SYNC").or_else(|_| env::var("DATABASE_URL")).unwrap_or_else(|_| {
        "postgres://sem_admin:CHANGE_ME_TO_A_STRONG_PASSWORD@localhost:5432/sem_swarm".to_string()
    });
    
    // Convert to sync URL if it contains asyncpg
    let db_url = db_url.replace("+asyncpg", "");

    info!("Connecting to database...");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&db_url)
        .await
        .context("Failed to connect to PostgreSQL")?;

    // 2. Fetch active facts
    info!("Fetching active epistemic facts...");
    let facts = sqlx::query_as::<_, EpistemicFact>(
        "SELECT id, fact_text, embedding, confidence_score 
         FROM epistemic_memory 
         WHERE is_active = true AND superseded_by IS NULL"
    )
    .fetch_all(&pool)
    .await?;

    let n = facts.len();
    info!("Loaded {} active facts into memory.", n);

    if n < 2 {
        info!("Not enough facts to consolidate. Sleeping.");
        return Ok(());
    }

    // Extract embeddings into a raw Vec<Vec<f32>> for sem-vector
    let embeddings: Vec<Vec<f32>> = facts.iter().map(|f| f.embedding.to_vec()).collect();

    // 3. Find Duplicates & Cluster
    info!("Running SIMD deduplication (threshold = 0.96)...");
    let dedup_start = Instant::now();
    let clusters = cluster_duplicates(&embeddings, 0.96);
    info!(
        "Found {} duplicate clusters to merge (took {:.2?}).",
        clusters.len(),
        dedup_start.elapsed()
    );

    // 4. Find Contradictions
    info!("Scanning for contradictions (topic > 0.70, dedup < 0.95)...");
    let contra_start = Instant::now();
    let contradictions = find_contradictions(&embeddings, 0.70, 0.95);
    info!(
        "Flagged {} potential contradictions (took {:.2?}).",
        contradictions.len(),
        contra_start.elapsed()
    );

#[derive(Serialize, Deserialize, Debug)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaResponse {
    response: String,
}

async fn consolidate_cluster(client: &reqwest::Client, ollama_url: &str, model: &str, texts: &[&str]) -> Result<String> {
    let mut prompt = String::from("Você é o processo de consolidação de memória do SEM-Swarm (Dreaming Loop).\nSua tarefa é fundir os seguintes fatos verificados redundantes em uma única frase coesa e clara, sem perda de informações importantes. Apenas responda com a frase consolidada, sem introduções.\n\n");
    for (i, t) in texts.iter().enumerate() {
        prompt.push_str(&format!("Fato {}: {}\n", i+1, t));
    }
    prompt.push_str("\n[FATO UNIFICADO]:");

    let req = OllamaRequest {
        model: model.to_string(),
        prompt,
        stream: false,
    };

    let res = client.post(&format!("{}/api/generate", ollama_url))
        .json(&req)
        .send()
        .await?
        .json::<OllamaResponse>()
        .await?;
        
    Ok(res.response.trim().to_string())
}

    // (Inside main, replace lines 101 to 114)
    // 5. Apply Consolidations to Database
    let client = reqwest::Client::new();
    let ollama_url = env::var("OLLAMA_VPS_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // Fallback to phi4-mini for local testing if deepseek is not available
    let deepseek_model = env::var("OLLAMA_DEEP_REASONING_MODEL").unwrap_or_else(|_| "phi4-mini".to_string());

    info!("Contacting Ollama at {} with model {} for consolidation...", ollama_url, deepseek_model);

    let mut consolidated_count = 0;
    
    for (i, cluster) in clusters.iter().enumerate() {
        let texts: Vec<&str> = cluster.iter().map(|&idx| facts[idx].fact_text.as_str()).collect();
        info!("Cluster {}: Consolidating {} facts...", i, cluster.len());
        
        match consolidate_cluster(&client, &ollama_url, &deepseek_model, &texts).await {
            Ok(unified_text) => {
                info!("  -> Consolidated fact: {}", unified_text);
                
                // 1. Calculate centroid embedding
                let dim = embeddings[0].len();
                let mut centroid = vec![0.0f32; dim];
                for &idx in cluster {
                    for d in 0..dim {
                        centroid[d] += embeddings[idx][d];
                    }
                }
                let n_f32 = cluster.len() as f32;
                for d in 0..dim {
                    centroid[d] /= n_f32;
                }
                
                // Normalize centroid
                let mut norm = 0.0;
                for d in 0..dim {
                    norm += centroid[d] * centroid[d];
                }
                norm = norm.sqrt();
                if norm > 0.0 {
                    for d in 0..dim {
                        centroid[d] /= norm;
                    }
                }
                
                // Convert centroid to string array for safe halfvec casting
                let centroid_str = format!("[{}]", centroid.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","));

                // 2. Begin transaction
                let mut tx = pool.begin().await?;
                
                // Insert new fact
                let new_fact_id: i64 = sqlx::query_scalar(
                    "INSERT INTO epistemic_memory (fact_text, embedding, confidence_score, metadata)
                     VALUES ($1, CAST($2 AS halfvec), $3, '{\"source\": \"dreaming_loop\"}')
                     RETURNING id"
                )
                .bind(&unified_text)
                .bind(&centroid_str)
                .bind(1.0)
                .fetch_one(&mut *tx)
                .await?;
                
                // Supersede old facts
                for &idx in cluster {
                    let old_id = facts[idx].id;
                    sqlx::query(
                        "UPDATE epistemic_memory SET is_active = false, superseded_by = $1 WHERE id = $2"
                    )
                    .bind(new_fact_id)
                    .bind(old_id)
                    .execute(&mut *tx)
                    .await?;
                }
                
                tx.commit().await?;
                consolidated_count += 1;
            }
            Err(e) => {
                error!("Failed to consolidate cluster {}: {:?}", i, e);
            }
        }
    }
    
    info!("Successfully consolidated {} clusters.", consolidated_count);

    info!(
        "✅ Dreaming Loop finished successfully in {:.2?}.",
        start_time.elapsed()
    );

    Ok(())
}
