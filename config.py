"""
config.py — Centralised settings loaded from environment variables.
"""
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3"
    ollama_embed_model: str = "nomic-embed-text"

    # --- Groq & Provider ---
    groq_api_key: str = ""
    groq_llm_model: str = "llama-3.3-70b-versatile"
    embedding_provider: str = "huggingface"

    # --- ChromaDB ---
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection_prefix: str = "codebase_"

    # --- LangSmith ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "codebase-qa-bot"

    # --- Git ---
    github_token: str = ""
    repos_dir: Path = Path("./data/repos")
    max_file_size_kb: int = 500
    max_repo_size_mb: int = 200

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- Retrieval ---
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
