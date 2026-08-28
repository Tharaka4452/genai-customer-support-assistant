import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer


class KnowledgeBase:
    """FAQ knowledge base with lexical TF-IDF, local LSA semantic retrieval, and hybrid RAG."""

    def __init__(self, csv_path: str, cache_dir: Optional[str] = None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path).fillna("")
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

        # Lexical baseline: word/character TF-IDF ensemble.
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

        # Local semantic layer: Latent Semantic Analysis (TF-IDF -> TruncatedSVD).
        # This produces dense concept vectors without consuming any external embedding API quota.
        requested_components = int(os.getenv("LSA_COMPONENTS", "64"))
        max_components = max(2, min(self.tfidf_matrix.shape[0] - 1, self.tfidf_matrix.shape[1] - 1))
        self.lsa_components = max(2, min(requested_components, max_components))
        self.lsa = TruncatedSVD(n_components=self.lsa_components, random_state=42)
        self.lsa_normalizer = Normalizer(copy=False)
        self.semantic_matrix = self.lsa_normalizer.fit_transform(
            self.lsa.fit_transform(self.tfidf_matrix)
        )

    @property
    def policy_count(self) -> int:
        return int(self.df["policy_id"].nunique())

    @property
    def has_api_key(self) -> bool:
        # The API key is only required for Gemini response generation, not retrieval.
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    def semantic_healthcheck(self) -> tuple[bool, str]:
        """Verify that the local dense semantic retriever is ready."""
        try:
            q = self._semantic_scores("customer support delivery question")
            ok = q.shape == (len(self.df),) and np.isfinite(q).all()
            return ok, f"Local semantic retriever OK: LSA ({self.lsa_components} dimensions)"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"[:500]

    def _lexical_scores(self, query: str) -> np.ndarray:
        word_full = cosine_similarity(self.vectorizer.transform([query]), self.tfidf_matrix).flatten()
        char_full = cosine_similarity(self.char_vectorizer.transform([query]), self.char_matrix).flatten()
        word_qk = cosine_similarity(self.qk_vectorizer.transform([query]), self.qk_matrix).flatten()
        char_qk = cosine_similarity(self.qk_char_vectorizer.transform([query]), self.qk_char_matrix).flatten()
        keyword = cosine_similarity(self.keyword_vectorizer.transform([query]), self.keyword_matrix).flatten()
        return (
            (0.20 * word_full) + (0.50 * char_full) + (0.10 * word_qk)
            + (0.10 * char_qk) + (0.10 * keyword)
        )

    def _semantic_scores(self, query: str) -> np.ndarray:
        q_tfidf = self.vectorizer.transform([query])
        q_dense = self.lsa.transform(q_tfidf)
        q_dense = self.lsa_normalizer.transform(q_dense)[0]
        return self.semantic_matrix @ q_dense

    @staticmethod
    def _rank_fusion(lexical_scores: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
        """Lexical-dominant reciprocal-rank fusion for stable hybrid retrieval."""
        n = len(lexical_scores)
        lexical_rank = np.empty(n, dtype=np.int32)
        semantic_rank = np.empty(n, dtype=np.int32)
        lexical_rank[np.argsort(lexical_scores)[::-1]] = np.arange(n)
        semantic_rank[np.argsort(semantic_scores)[::-1]] = np.arange(n)
        # Keep the benchmark-stable lexical retriever dominant while allowing dense semantic
        # evidence to break ties and improve paraphrase robustness.
        return (20.0 / (21.0 + lexical_rank)) + (1.0 / (21.0 + semantic_rank))

    def _collapse_results(
        self,
        ranking_scores: np.ndarray,
        lexical_scores: np.ndarray,
        semantic_scores: Optional[np.ndarray],
        top_k: int,
        retrieval_mode: str,
        confidence_scores: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        best_by_policy: Dict[str, int] = {}
        for idx in np.argsort(ranking_scores)[::-1]:
            policy_id = str(self.df.iloc[idx]["policy_id"])
            if policy_id not in best_by_policy:
                best_by_policy[policy_id] = int(idx)
            if len(best_by_policy) >= max(top_k, 1):
                break

        scores_for_display = confidence_scores if confidence_scores is not None else ranking_scores
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
                "score": round(float(scores_for_display[idx]), 4),
                "lexical_score": round(float(lexical_scores[idx]), 4),
                "semantic_score": round(sem, 4) if sem is not None else None,
                "retrieval_mode": retrieval_mode,
            })
        return results

    def search(self, query: str, top_k: int = 3, mode: str = "auto") -> List[Dict]:
        query = (query or "").strip()
        if not query:
            return []

        lexical = self._lexical_scores(query)
        self.last_retrieval_error = ""
        requested = (mode or "auto").strip().lower()

        if requested in {"tfidf", "baseline", "lexical"}:
            return self._collapse_results(
                lexical, lexical, None, top_k, "TF-IDF lexical baseline", confidence_scores=lexical
            )

        try:
            semantic = self._semantic_scores(query)
        except Exception as exc:
            self.last_retrieval_error = f"{type(exc).__name__}: {exc}"[:500]
            return self._collapse_results(
                lexical, lexical, None, top_k, "TF-IDF fallback", confidence_scores=lexical
            )

        if requested in {"semantic", "lsa", "embeddings"}:
            # Map cosine similarity from roughly -1..1 into 0..1 for a readable confidence score.
            semantic_conf = np.clip((semantic + 1.0) / 2.0, 0.0, 1.0)
            return self._collapse_results(
                semantic, lexical, semantic, top_k,
                f"Local semantic LSA ({self.lsa_components}D)", confidence_scores=semantic_conf
            )

        # Auto and Hybrid are fully local for retrieval; Gemini is only used after retrieval
        # to generate the grounded customer-facing answer.
        fused = self._rank_fusion(lexical, semantic)
        return self._collapse_results(
            fused, lexical, semantic, top_k,
            f"Hybrid: TF-IDF + local LSA ({self.lsa_components}D)",
            confidence_scores=lexical,
        )
