"""Pre-compute and cache FAQ embeddings. Requires GEMINI_API_KEY."""
from pathlib import Path

from dotenv import load_dotenv

from src.knowledge_base import KnowledgeBase

load_dotenv()
base = Path(__file__).parent
kb = KnowledgeBase(str(base / "data" / "faq.csv"))
try:
    matrix = kb._ensure_embedding_matrix()
    print(
        f"Embedding cache ready: {matrix.shape[0]} vectors x {matrix.shape[1]} dimensions "
        f"using {kb.embedding_model}"
    )
except Exception as exc:
    raise SystemExit(f"Could not build embeddings: {exc}")
