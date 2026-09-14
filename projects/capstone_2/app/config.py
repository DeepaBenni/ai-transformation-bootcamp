"""Application settings.

One Settings object, constructed once, cached. No other module may call os.getenv
or os.environ: if a value is configurable, it belongs here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory that holds .env and this app/ package.
# app/config.py -> parent is app/, parent.parent is projects/capstone_2.
ROOT: Path = Path(__file__).resolve().parent.parent

# Repo root = two levels above the project root (projects/capstone_2/ -> repo).
# Shared corpora live in the repo-wide data/ folder, not inside this project.
REPO_ROOT: Path = ROOT.parent.parent


class Settings(BaseSettings):
    """Runtime configuration, loaded from .env and the process environment."""

    model_config = SettingsConfigDict(
        # Repo-root .env first, project-local .env second: a value set here in
        # projects/capstone_2/.env overrides the shared one at the repo root.
        env_file=(REPO_ROOT / ".env", ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds keys other tools use; ignore them
        case_sensitive=False,  # MYSQL_URL in .env maps to mysql_url here
    )

    # ---- models ------------------------------------------------------------
    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    temperature: float = 0.0

    # ---- datastores --------------------------------------------------------
    mysql_url: str = "mysql+pymysql://opsmate:opsmate@127.0.0.1:3306/opsmate"
    chroma_dir: Path = ROOT / "chroma_opsmate"
    kb_collection: str = "kb_articles"
    kb_source_dir: Path = REPO_ROOT / "data" / "capstone_2_knowledge_base"

    # ---- retrieval ---------------------------------------------------------
    retrieval_k: int = 5
    relevance_floor: float = 0.35
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- agent limits ------------------------------------------------------
    max_hops: int = 8
    recursion_limit: int = 30
    tool_timeout_s: float = 5.0

    # ---- cost ------------------------------------------------------------------
    # USD per 1,000 tokens. VERIFY against current OpenAI pricing before you
    # publish a cost figure in the R8 report - a stale constant silently
    # falsifies an honest number.
    price_input_per_1k: float = 0.00015
    price_output_per_1k: float = 0.00060

    # ---- observability -----------------------------------------------------
    trace_dir: Path = ROOT / "traces"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Cached so the .env file is parsed once per process and every module observes
    identical configuration.
    """
    settings = Settings()
    # Create the write-target directories eagerly so first use never fails on a
    # missing folder.
    settings.trace_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
