from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


def _as_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if mat.ndim == 1:
        norm = float(np.linalg.norm(mat))
        if norm < eps:
            norm = eps
        return mat / norm

    if mat.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array for normalization, got shape={mat.shape}")

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


@dataclass
class BgeM3Embedder:
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 64
    normalize_embeddings: bool = True
    expected_dim: Optional[int] = None

    _model: Optional[SentenceTransformer] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        dev = str(self.device).strip().lower()
        self.device = "cuda" if dev == "cuda" else "cpu"

        if int(self.batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        self.batch_size = int(self.batch_size)

        if self.expected_dim is not None:
            self.expected_dim = int(self.expected_dim)

    def load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def model(self) -> SentenceTransformer:
        self.load()
        assert self._model is not None
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")

        texts_list = [str(t) if t is not None else "" for t in texts]
        if len(texts_list) == 0:
            if self.expected_dim is None:
                raise ValueError("Cannot embed an empty sequence when expected_dim is unknown")
            return np.zeros((0, self.expected_dim), dtype=np.float32)

        arr = self.model.encode(
            texts_list,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        arr = _as_float32(arr)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape={arr.shape}")

        if self.normalize_embeddings:
            arr = _l2_normalize(arr)

        self._validate_dim(arr.shape[1])
        return arr

    def embed_text(self, text: str) -> np.ndarray:
        vec = self.embed_texts([text])[0]
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D embedding vector, got shape={vec.shape}")
        return vec

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_text(text)

    def dim(self) -> int:
        if self.expected_dim is not None:
            return self.expected_dim

        test_vec = self.embed_query("test")
        dim = int(test_vec.shape[0])
        self.expected_dim = dim
        return dim

    def _validate_dim(self, actual_dim: int) -> None:
        if self.expected_dim is None:
            return
        if int(actual_dim) != int(self.expected_dim):
            raise ValueError(
                f"Embedding dim mismatch: actual={actual_dim}, expected={self.expected_dim}"
            )