"""
SEM-Swarm — API Configuration
Loads settings from environment variables with sensible defaults.

Architecture follows the "Cognitive Minimum Privilege" principle:
each model is assigned exactly the capacity its task requires.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    # PostgreSQL
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "sem_swarm"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "sem_admin"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "changeme"))

    # API
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_log_level: str = field(default_factory=lambda: os.getenv("API_LOG_LEVEL", "info"))

    # ── Ollama Endpoints ─────────────────────────────────────
    ollama_local_url: str = field(default_factory=lambda: os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434"))
    ollama_vps_url: str = field(default_factory=lambda: os.getenv("OLLAMA_VPS_URL", ""))

    # ── Model Assignments (Cognitive Minimum Privilege) ──────
    # Orchestrator: Phi-4-Mini (3.8B) — tool calling, MCP, chat
    orchestrator_model: str = field(default_factory=lambda: os.getenv("OLLAMA_ORCHESTRATOR_MODEL", "phi4-mini"))
    # Extractor: NuExtract (3.8B) — deterministic JSON extraction
    extractor_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EXTRACTOR_MODEL", "nuextract"))
    # FIM Copilot: Qwen2.5-Coder (1.5B) — autocomplete only
    fim_model: str = field(default_factory=lambda: os.getenv("OLLAMA_FIM_MODEL", "qwen2.5-coder:1.5b"))
    # Reasoning: Qwen3 (8B) — complex logic on-demand
    reasoning_model: str = field(default_factory=lambda: os.getenv("OLLAMA_REASONING_MODEL", "qwen3:8b"))
    # Embeddings: qwen3-embedding (VPS) — multilingual 100+ languages
    embed_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding"))
    # Deep Reasoning: DeepSeek-R1 14B (VPS) — async dreaming loop
    deep_reasoning_model: str = field(default_factory=lambda: os.getenv("OLLAMA_DEEP_REASONING_MODEL", "deepseek-r1:14b"))

    # Embedding dimension (must match VECTOR column in init.sql)
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "2048")))

    @property
    def database_url(self) -> str:
        """Async database URL for asyncpg."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """Sync database URL (for migrations/scripts)."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def embedding_endpoint(self) -> str:
        """The Ollama endpoint for embeddings (VPS if configured, else local)."""
        return self.ollama_vps_url if self.ollama_vps_url else self.ollama_local_url


settings = Settings()
