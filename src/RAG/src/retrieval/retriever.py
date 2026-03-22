from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..artifacts.loader import ArtifactsStore, SectionMeta, ChunkMeta
from .embedder import BgeM3Embedder
from .faiss_store import SectionsFaissIndex, ChunksFaissIndex, FaissSearchHit


@dataclass(frozen=True)
class RetrievedSection:
    doc_id: str
    section_id: str
    title: str
    start_page: int
    end_page: int
    summary: str
    vector_score: float
    is_fallback_window: bool = False


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    section_id: str
    chunk_id: str
    section_title: str
    page_start: int
    page_end: int
    length_tokens: int
    vector_score: float


@dataclass(frozen=True)
class RetrievedPage:
    doc_id: str
    page_no: int
    text: str
    vector_score: float


@dataclass(frozen=True)
class RetrievalBundle:
    query: str
    retrieved_sections: List[RetrievedSection]
    retrieved_chunks: List[RetrievedChunk]
    retrieved_pages: List[RetrievedPage]
    primary_doc_id: Optional[str]


def _best_score_by_key(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for key, score in pairs:
        if key not in best or score > best[key]:
            best[key] = score
    return best


def _dedup_sections_keep_best(items: List[RetrievedSection]) -> List[RetrievedSection]:
    best: Dict[str, RetrievedSection] = {}
    ordered_ids: List[str] = []

    for item in items:
        sid = item.section_id
        if sid not in best:
            best[sid] = item
            ordered_ids.append(sid)
        elif item.vector_score > best[sid].vector_score:
            best[sid] = item

    return [best[sid] for sid in ordered_ids]


def _dedup_chunks_keep_best(items: List[RetrievedChunk]) -> List[RetrievedChunk]:
    best: Dict[str, RetrievedChunk] = {}
    ordered_ids: List[str] = []

    for item in items:
        cid = item.chunk_id
        if cid not in best:
            best[cid] = item
            ordered_ids.append(cid)
        elif item.vector_score > best[cid].vector_score:
            best[cid] = item

    return [best[cid] for cid in ordered_ids]


def _dedup_pages_keep_best(pairs: List[Tuple[Tuple[str, int], float]]) -> List[Tuple[Tuple[str, int], float]]:
    best: Dict[Tuple[str, int], float] = {}
    ordered_keys: List[Tuple[str, int]] = []

    for key, score in pairs:
        if key not in best:
            best[key] = score
            ordered_keys.append(key)
        elif score > best[key]:
            best[key] = score

    return [(key, best[key]) for key in ordered_keys]


@dataclass
class HierarchicalPageRetriever:
    artifacts: ArtifactsStore
    embedder: BgeM3Embedder

    top_k_sections: int = 8
    top_k_chunks_raw: int = 120
    max_chunks_after_filter: int = 24
    max_pages: int = 25
    neighbors: int = 0
    drop_empty_pages: bool = True
    restrict_to_primary_doc: bool = True

    _sections_index: Optional[SectionsFaissIndex] = None
    _chunks_index: Optional[ChunksFaissIndex] = None

    def __post_init__(self) -> None:
        self.top_k_sections = int(self.top_k_sections)
        self.top_k_chunks_raw = int(self.top_k_chunks_raw)
        self.max_chunks_after_filter = int(self.max_chunks_after_filter)
        self.max_pages = int(self.max_pages)
        self.neighbors = int(self.neighbors)

        if self.top_k_sections <= 0:
            raise ValueError("top_k_sections must be positive")
        if self.top_k_chunks_raw <= 0:
            raise ValueError("top_k_chunks_raw must be positive")
        if self.max_chunks_after_filter <= 0:
            raise ValueError("max_chunks_after_filter must be positive")
        if self.max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if self.neighbors < 0:
            raise ValueError("neighbors must be >= 0")

    @property
    def sections_index(self) -> SectionsFaissIndex:
        if self._sections_index is None:
            self._sections_index = SectionsFaissIndex(self.artifacts)
        return self._sections_index

    @property
    def chunks_index(self) -> ChunksFaissIndex:
        if self._chunks_index is None:
            self._chunks_index = ChunksFaissIndex(self.artifacts)
        return self._chunks_index

    def retrieve(self, question_text: str) -> RetrievalBundle:
        qvec = self.embedder.embed_query(question_text)

        retrieved_sections = self._retrieve_sections(qvec)
        selected_section_ids = {item.section_id for item in retrieved_sections}

        retrieved_chunks = self._retrieve_chunks(
            qvec=qvec,
            allowed_section_ids=selected_section_ids,
        )

        primary_doc_id = self.select_primary_doc(
            sections=retrieved_sections,
            chunks=retrieved_chunks,
        )

        retrieved_pages = self._retrieve_parent_pages(
            chunks=retrieved_chunks,
            primary_doc_id=primary_doc_id if self.restrict_to_primary_doc else None,
        )

        if self.restrict_to_primary_doc and primary_doc_id is not None:
            retrieved_sections = [x for x in retrieved_sections if x.doc_id == primary_doc_id]
            retrieved_chunks = [x for x in retrieved_chunks if x.doc_id == primary_doc_id]

        return RetrievalBundle(
            query=question_text,
            retrieved_sections=retrieved_sections,
            retrieved_chunks=retrieved_chunks,
            retrieved_pages=retrieved_pages,
            primary_doc_id=primary_doc_id,
        )

    def retrieve_pages(self, question_text: str) -> List[RetrievedPage]:
        bundle = self.retrieve(question_text)
        return bundle.retrieved_pages

    def _retrieve_sections(self, qvec) -> List[RetrievedSection]:
        hits: List[FaissSearchHit[SectionMeta]] = self.sections_index.search_hits(
            query_vec=qvec,
            top_k=self.top_k_sections,
        )

        out: List[RetrievedSection] = []
        for hit in hits:
            meta = hit.meta
            out.append(
                RetrievedSection(
                    doc_id=meta.doc_id,
                    section_id=meta.section_id,
                    title=meta.title,
                    start_page=meta.start_page,
                    end_page=meta.end_page,
                    summary=meta.summary,
                    vector_score=float(hit.score),
                    is_fallback_window=bool(meta.is_fallback_window),
                )
            )

        return _dedup_sections_keep_best(out)

    def _retrieve_chunks(
        self,
        qvec,
        allowed_section_ids: Set[str],
    ) -> List[RetrievedChunk]:
        hits: List[FaissSearchHit[ChunkMeta]] = self.chunks_index.search_hits(
            query_vec=qvec,
            top_k=self.top_k_chunks_raw,
        )

        filtered: List[RetrievedChunk] = []
        for hit in hits:
            meta = hit.meta
            if allowed_section_ids and meta.section_id not in allowed_section_ids:
                continue

            filtered.append(
                RetrievedChunk(
                    doc_id=meta.doc_id,
                    section_id=meta.section_id,
                    chunk_id=meta.chunk_id,
                    section_title=meta.section_title,
                    page_start=meta.page_start,
                    page_end=meta.page_end,
                    length_tokens=meta.length_tokens,
                    vector_score=float(hit.score),
                )
            )

        filtered = _dedup_chunks_keep_best(filtered)
        return filtered[: self.max_chunks_after_filter]

    def _retrieve_parent_pages(
        self,
        chunks: List[RetrievedChunk],
        primary_doc_id: Optional[str] = None,
    ) -> List[RetrievedPage]:
        page_pairs: List[Tuple[Tuple[str, int], float]] = []

        for chunk in chunks:
            if primary_doc_id is not None and chunk.doc_id != primary_doc_id:
                continue

            for page_no in range(int(chunk.page_start), int(chunk.page_end) + 1):
                page_pairs.append(((chunk.doc_id, page_no), float(chunk.vector_score)))

                if self.neighbors > 0:
                    for delta in range(1, self.neighbors + 1):
                        prev_page = page_no - delta
                        next_page = page_no + delta

                        if prev_page >= 1:
                            page_pairs.append(((chunk.doc_id, prev_page), float(chunk.vector_score) * 0.999))
                        page_pairs.append(((chunk.doc_id, next_page), float(chunk.vector_score) * 0.999))

        page_pairs = _dedup_pages_keep_best(page_pairs)

        page_pairs.sort(key=lambda x: x[1], reverse=True)
        page_pairs = page_pairs[: self.max_pages]

        pages_out: List[RetrievedPage] = []
        seen: Set[Tuple[str, int]] = set()

        for (doc_id, page_no), score in page_pairs:
            key = (doc_id, page_no)
            if key in seen:
                continue
            seen.add(key)

            text = self.artifacts.get_page_text(doc_id, page_no)
            if self.drop_empty_pages and (not text or not text.strip()):
                continue

            pages_out.append(
                RetrievedPage(
                    doc_id=doc_id,
                    page_no=page_no,
                    text=text,
                    vector_score=float(score),
                )
            )

        return pages_out

    def select_primary_doc(
        self,
        sections: List[RetrievedSection],
        chunks: List[RetrievedChunk],
    ) -> Optional[str]:
        scores: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for item in sections:
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + float(item.vector_score) * 2.0
            counts[item.doc_id] = counts.get(item.doc_id, 0) + 1

        for item in chunks:
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + float(item.vector_score)
            counts[item.doc_id] = counts.get(item.doc_id, 0) + 1

        if not scores:
            return None

        ranked = sorted(
            scores.keys(),
            key=lambda doc_id: (scores[doc_id], counts.get(doc_id, 0)),
            reverse=True,
        )
        return ranked[0]


# Backward-compatible alias for the rest of the pipeline
PageRetriever = HierarchicalPageRetriever