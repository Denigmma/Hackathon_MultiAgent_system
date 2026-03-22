from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Sequence, Tuple, TypeVar

import numpy as np

from ..artifacts.loader import ArtifactsStore, SectionMeta, ChunkMeta


MetaT = TypeVar("MetaT", SectionMeta, ChunkMeta)


def _prepare_query(query_vec: np.ndarray) -> np.ndarray:
    if not isinstance(query_vec, np.ndarray):
        query_vec = np.asarray(query_vec, dtype=np.float32)

    if query_vec.dtype != np.float32:
        query_vec = query_vec.astype(np.float32, copy=False)

    if query_vec.ndim != 1:
        raise ValueError("query_vec must be a 1D vector")

    return query_vec.reshape(1, -1)


@dataclass(frozen=True)
class FaissSearchHit(Generic[MetaT]):
    index_pos: int
    score: float
    meta: MetaT


class _BaseGlobalFaissIndex(Generic[MetaT]):
    def __init__(self, artifacts: ArtifactsStore, index_name: str):
        self.artifacts = artifacts
        self.index_name = index_name

    def _load_index(self):
        raise NotImplementedError

    def _load_meta(self) -> Sequence[MetaT]:
        raise NotImplementedError

    def _get_meta_by_index_pos(self, index_pos: int) -> MetaT:
        raise NotImplementedError

    def search(self, query_vec: np.ndarray, top_k: int = 30) -> Tuple[List[int], List[float]]:
        index = self._load_index()
        q = _prepare_query(query_vec)

        scores, idxs = index.search(q, int(top_k))
        idx_list = idxs[0].tolist()
        score_list = scores[0].tolist()

        out_idx: List[int] = []
        out_scores: List[float] = []

        for i, s in zip(idx_list, score_list):
            if int(i) < 0:
                continue
            out_idx.append(int(i))
            out_scores.append(float(s))

        return out_idx, out_scores

    def search_hits(self, query_vec: np.ndarray, top_k: int = 30) -> List[FaissSearchHit[MetaT]]:
        idxs, scores = self.search(query_vec=query_vec, top_k=top_k)
        hits: List[FaissSearchHit[MetaT]] = []

        for index_pos, score in zip(idxs, scores):
            meta = self._get_meta_by_index_pos(index_pos)
            hits.append(
                FaissSearchHit(
                    index_pos=index_pos,
                    score=float(score),
                    meta=meta,
                )
            )

        return hits

    def load_meta(self) -> List[MetaT]:
        return list(self._load_meta())


class SectionsFaissIndex(_BaseGlobalFaissIndex[SectionMeta]):
    def __init__(self, artifacts: ArtifactsStore):
        super().__init__(artifacts=artifacts, index_name="sections")

    def _load_index(self):
        return self.artifacts.load_sections_faiss_index()

    def _load_meta(self) -> Sequence[SectionMeta]:
        return self.artifacts.load_sections_meta()

    def _get_meta_by_index_pos(self, index_pos: int) -> SectionMeta:
        return self.artifacts.get_section_meta_by_index_pos(index_pos)


class ChunksFaissIndex(_BaseGlobalFaissIndex[ChunkMeta]):
    def __init__(self, artifacts: ArtifactsStore):
        super().__init__(artifacts=artifacts, index_name="chunks")

    def _load_index(self):
        return self.artifacts.load_chunks_faiss_index()

    def _load_meta(self) -> Sequence[ChunkMeta]:
        return self.artifacts.load_chunks_meta()

    def _get_meta_by_index_pos(self, index_pos: int) -> ChunkMeta:
        return self.artifacts.get_chunk_meta_by_index_pos(index_pos)