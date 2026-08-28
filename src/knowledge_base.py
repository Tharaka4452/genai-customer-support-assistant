import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:
    """FAQ knowledge base with a TF-IDF baseline and optional Gemini semantic retrieval."""

    def __init__(self, csv_path: str, cache_dir: Optional[str] = None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path).fillna("")
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.embedding_dimension = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", "768"))
        self.last_retrieval_error = ""

        required = {
            "faq_id", "policy_id", "category", "intent", "question", "answer",
            "keywords", "priority", "escalation_required", "risk_type"
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

        self.df["search_text"] = (
            "Question: " + self.df["question"].astype(str)
            + " Answer: " + self.df["answer"].astype(str)
            + " Keywords: " + self.df["keywords"].astype(str)
            + " Category: " + self.df["category"].astype(str)
            + " Intent: " + self.df["intent"].astype(str)
        )

        # Lexical baseline: word/character TF-IDF ensemble. Character n-grams make
        # the baseline more tolerant of spelling and wording changes while remaining
        # a purely lexical method that can be compared against semantic embeddings.
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["search_text"])

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        self.char_matrix = self.char_vectorizer.fit_transform(self.df["search_text"])

        self.qk_text = self.df["question"].astype(str) + " " + self.df["keywords"].astype(str)
        self.qk_vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.qk_matrix = self.qk_vectorizer.fit_transform(self.qk_text)
        self.qk_char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        self.qk_char_matrix = self.qk_char_vectorizer.fit_transform(self.qk_text)

        self.keyword_text = self.df["keywords"].astype(str) + " " + self.df["category"].astype(str)
        self.keyword_vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.keyword_matrix = self.keyword_vectorizer.fit_transform(self.keyword_text)

        self.cache_dir = Path(cache_dir) if cache_dir else self.csv_path.parent.parent / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_matrix = None

    @property
    def policy_count(self) -> int:
        return int(self.df["policy_id"].nunique())

    @property
    def has_api_key(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    def _cache_path(self) -> Path:
        payload = (
            "\n".join(self.df["search_text"].astype(str))
            + self.embedding_model
            + str(self.embedding_dimension)
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return self.cache_dir / f"faq_embeddings_{digest}.npz"

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _safe_error(self, exc: Exception) -> str:
        """Return a useful diagnostic without ever exposing the API key."""
        message = f"{type(exc).__name__}: {exc}"
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        # Redact anything that looks like a Google API key as a second safety net.
        message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED_API_KEY]", message)
        return message[:700]

    def _embed_texts(self, texts: List[str], task_type: str) -> np.ndarray:
        from google import genai
        from google.genai import types

        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.embed_content(
            model=self.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=self.embedding_dimension,
                task_type=task_type,
            ),
        )
        vectors = [item.values for item in (response.embeddings or [])]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Gemini returned {len(vectors)} embeddings for {len(texts)} input texts."
            )
        return np.asarray(vectors, dtype=np.float32)

    def embedding_healthcheck(self) -> tuple[bool, str]:
        """Make one tiny embedding request so deployment issues are easy to diagnose."""
        if not self.has_api_key:
            return False, "GEMINI_API_KEY is not configured."
        try:
            vector = self._embed_texts(["customer support delivery question"], "RETRIEVAL_QUERY")
            return (
                vector.shape == (1, self.embedding_dimension),
                f"Embedding API OK: {self.embedding_model} ({vector.shape[1]} dimensions)",
            )
        except Exception as exc:
            return False, self._safe_error(exc)

    def _ensure_embedding_matrix(self) -> np.ndarray:
        if self._embedding_matrix is not None:
            return self._embedding_matrix

        cache_path = self._cache_path()
        if cache_path.exists():
            self._embedding_matrix = np.load(cache_path)["embeddings"]
            return self._embedding_matrix

        if not self.has_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured for semantic retrieval.")

        texts = self.df["search_text"].astype(str).tolist()
        vectors: List[np.ndarray] = []
        # Keep requests small enough for hosted deployments and conservative API payload limits.
        # 25 FAQ records are ~9 KB or less with this dataset, while still avoiding excessive calls.
        batch_size = int(os.getenv("GEMINI_EMBEDDING_BATCH_SIZE", "25"))
        batch_size = max(1, min(batch_size, 25))
        for start in range(0, len(texts), batch_size):
            vectors.append(
                self._embed_texts(texts[start:start + batch_size], "RETRIEVAL_DOCUMENT")
            )

        matrix = self._normalize_rows(np.vstack(vectors))
        np.savez_compressed(cache_path, embeddings=matrix)
        self._embedding_matrix = matrix
        return matrix

    def _semantic_scores(self, query: str) -> np.ndarray:
        matrix = self._ensure_embedding_matrix()
        q = self._embed_texts([query], "RETRIEVAL_QUERY")
        q = self._normalize_rows(q)[0]
        return matrix @ q

    def _collapse_results(
        self,
        final_scores: np.ndarray,
        lexical_scores: np.ndarray,
        semantic_scores: Optional[np.ndarray],
        top_k: int,
        retrieval_mode: str,
    ) -> List[Dict]:
        best_by_policy: Dict[str, int] = {}
        for idx in np.argsort(final_scores)[::-1]:
            policy_id = str(self.df.iloc[idx]["policy_id"])
            if policy_id not in best_by_policy:
                best_by_policy[policy_id] = int(idx)
            if len(best_by_policy) >= max(top_k, 1):
                break

        results = []
        for idx in best_by_policy.values():
            row = self.df.iloc[idx]
            sem = float(semantic_scores[idx]) if semantic_scores is not None else None
            results.append({
                "faq_id": str(row["faq_id"]),
                "policy_id": str(row["policy_id"]),
                "category": str(row["category"]),
                "intent": str(row["intent"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "priority": str(row["priority"]),
                "escalation_required": str(row["escalation_required"]),
                "risk_type": str(row["risk_type"]),
                "score": round(float(final_scores[idx]), 4),
                "lexical_score": round(float(lexical_scores[idx]), 4),
                "semantic_score": round(sem, 4) if sem is not None else None,
                "retrieval_mode": retrieval_mode,
            })
        return results

    def search(self, query: str, top_k: int = 3, mode: str = "auto") -> List[Dict]:
        query = (query or "").strip()
        if not query:
            return []

        word_full = cosine_similarity(self.vectorizer.transform([query]), self.tfidf_matrix).flatten()
        char_full = cosine_similarity(self.char_vectorizer.transform([query]), self.char_matrix).flatten()
        word_qk = cosine_similarity(self.qk_vectorizer.transform([query]), self.qk_matrix).flatten()
        char_qk = cosine_similarity(self.qk_char_vectorizer.transform([query]), self.qk_char_matrix).flatten()
        keyword = cosine_similarity(self.keyword_vectorizer.transform([query]), self.keyword_matrix).flatten()
        lexical = (
            (0.20 * word_full) + (0.50 * char_full) + (0.10 * word_qk)
            + (0.10 * char_qk) + (0.10 * keyword)
        )
        self.last_retrieval_error = ""

        requested = (mode or "auto").strip().lower()
        if requested in {"tfidf", "baseline", "lexical"}:
            return self._collapse_results(lexical, lexical, None, top_k, "TF-IDF baseline")

        if requested in {"semantic", "embeddings"} or (requested in {"auto", "hybrid"} and self.has_api_key):
            try:
                semantic = self._semantic_scores(query)
                if requested in {"semantic", "embeddings"}:
                    # Gemini embeddings are cosine-scored after L2 normalization. Map -1..1 to 0..1 for display.
                    final = np.clip((semantic + 1.0) / 2.0, 0.0, 1.0)
                    return self._collapse_results(final, lexical, semantic, top_k, "Gemini embeddings")

                lexical_scaled = np.clip(lexical, 0.0, 1.0)
                semantic_scaled = np.clip((semantic + 1.0) / 2.0, 0.0, 1.0)
                final = (0.40 * lexical_scaled) + (0.60 * semantic_scaled)
                return self._collapse_results(final, lexical, semantic, top_k, "Hybrid: TF-IDF + Gemini embeddings")
            except Exception as exc:
                self.last_retrieval_error = self._safe_error(exc)
                return self._collapse_results(lexical, lexical, None, top_k, "TF-IDF fallback")

        return self._collapse_results(lexical, lexical, None, top_k, "TF-IDF baseline")
