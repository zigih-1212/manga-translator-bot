import re
from pathlib import Path

import numpy as np

from cfg import DATA_DIR
from .log import log

_EMBED_DIM = 512
_NGRAM = 3


class RAGIndex:
    """Lightweight retrieval index over prior translations.

    Uses character n-gram feature hashing (pure numpy, no torch/faiss needed).
    Optional: if `faiss` is installed, uses IndexFlatIP for faster search.
    """

    def __init__(self, dim: int = _EMBED_DIM, ngram: int = _NGRAM):
        self.dim = dim
        self.ngram = ngram
        self._texts: list[str] = []
        self._matrix: np.ndarray | None = None
        self._faiss_index = None
        self._index = None
        self._try_faiss()

    def _try_faiss(self):
        try:
            import faiss  # type: ignore
            self._faiss_index = faiss.IndexFlatIP(self.dim)
            log.info("FAISS available, using IndexFlatIP")
        except Exception:
            self._faiss_index = None

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _hash_ngrams(self, text: str) -> np.ndarray:
        text = self._normalize(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec
        grams = set()
        for i in range(len(text) - self.ngram + 1):
            grams.add(text[i:i + self.ngram])
        # Also include 1-2 grams for short words
        for n in (1, 2):
            for i in range(len(text) - n + 1):
                grams.add(text[i:i + n])
        for g in grams:
            h = hash(g) % self.dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def rebuild(self, texts: list[str]):
        """Rebuild index from a list of source-language strings."""
        self._texts = list(texts)
        if not self._texts:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            return
        rows = [self._hash_ngrams(t) for t in self._texts]
        self._matrix = np.vstack(rows)
        if self._faiss_index is not None:
            self._faiss_index.reset()
            self._faiss_index.add(self._matrix)

    def add(self, text: str):
        if not text or not text.strip():
            return
        vec = self._hash_ngrams(text)
        if self._matrix is None:
            self._matrix = vec[None, :]
        else:
            self._matrix = np.vstack([self._matrix, vec[None, :]])
        if self._faiss_index is not None:
            self._faiss_index.add(vec[None, :])
        self._texts.append(text)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not query or self._texts is None or len(self._texts) == 0:
            return []
        q = self._hash_ngrams(query)
        k = min(k, len(self._texts))
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q[None, :], k)
            results = []
            for i, score in zip(idxs[0], scores[0]):
                if int(i) < 0 or int(i) >= len(self._texts):
                    continue
                results.append((self._texts[int(i)], float(score)))
            return results
        dots = self._matrix @ q
        top = np.argsort(dots)[::-1][:k]
        return [(self._texts[i], float(dots[i])) for i in top if dots[i] > 0.25]


def load_translations_from_memory(manga_id: str) -> list[str]:
    """Collect all previously translated source texts for a manga."""
    import json
    try:
        with open(DATA_DIR / "memory.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    entry = data.get(manga_id, {})
    chapters = entry.get("chapters", {})
    texts = []
    for ch in chapters.values():
        for p in ch:
            ko = (p.get("ko") or "").strip()
            if ko:
                texts.append(ko)
    return texts
