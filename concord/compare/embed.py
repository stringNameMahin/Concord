"""Local sentence embeddings, used only to propose candidate pairs.

Two properties matter here and neither is retrieval quality. First, no API key:
the brief requires a reviewer to evaluate the project without our account, so
the embedding path stays on the machine. Second, embeddings never decide
anything - they widen the candidate set that the deterministic decision table
then judges. "Revenue was Rs 8,142 Cr" and "Revenue was Rs 7,224 Cr" sit right
next to each other in embedding space, which is exactly why cosine similarity
is a recall device here and never a verdict.
"""

from __future__ import annotations

import numpy as np

from concord import config

_MODEL = None


def load_model(name: str | None = None):
    """Load the sentence-transformer once per process.

    Imported lazily so that importing the comparison engine does not drag in
    torch, and so the rest of the test suite runs without the model present.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(name or config.EMBED_MODEL)
    return _MODEL


class LocalEncoder:
    """Encoder protocol: `encode(list[str]) -> (n, d) float32, rows L2-normed."""

    def __init__(self, name: str | None = None, batch_size: int = 64):
        self.name = name or config.EMBED_MODEL
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = load_model(self.name).encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def top_k(matrix: np.ndarray, k: int, min_similarity: float = 0.0) -> list[list[int]]:
    """Brute-force cosine neighbours, k per row, self excluded.

    At a few thousand facts the full similarity matrix is a few tens of MB and
    the whole search is milliseconds, so there is no vector database here and
    no index to keep in sync with the ledger.
    """
    n = matrix.shape[0]
    if n < 2 or k < 1:
        return [[] for _ in range(n)]

    rows = unit_rows(matrix.astype(np.float32))
    similarity = rows @ rows.T
    np.fill_diagonal(similarity, -np.inf)

    width = min(k, n - 1)
    partition = np.argpartition(-similarity, width - 1, axis=1)[:, :width]

    neighbours: list[list[int]] = []
    for row in range(n):
        candidates = partition[row]
        scores = similarity[row, candidates]
        order = np.argsort(-scores)
        neighbours.append(
            [int(candidates[i]) for i in order if scores[i] >= min_similarity]
        )
    return neighbours
