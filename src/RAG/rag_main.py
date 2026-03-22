from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

from .src.artifacts.loader import ArtifactsStore
from .src.retrieval.embedder import BgeM3Embedder
from .src.retrieval.retriever import HierarchicalPageRetriever, RetrievalBundle, RetrievedPage
from .src.rerank.reranker import PageReranker, RankedPage
from .src.answering.answerer import RAGAnswerer, AnswerResult

from .src.rerank.client import ModelClient as RerankModelClient
from .src.answering.client import ModelClient as AnswerModelClient


# -----------------------------------------------------------------------------
# ENV / CONFIG
# -----------------------------------------------------------------------------

def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _env_str(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return str(val)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RAGSettings:
    artifacts_root: str

    openrouter_api_key: str
    openrouter_base_url: str

    rerank_model: str
    answer_model: str

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    top_k_sections: int = 8
    top_k_chunks_raw: int = 120
    max_chunks_after_filter: int = 24
    max_pages: int = 25
    neighbors: int = 0
    restrict_to_primary_doc: bool = True

    max_pages_for_rerank: int = 12
    max_pages_for_answer: int = 6
    max_chars_per_page_for_answer: int = 12000

    drop_empty_pages: bool = True

    @staticmethod
    def from_env() -> "RAGSettings":
        _load_env()

        artifacts_root = os.getenv("ARTIFACTS_ROOT", str(Path(__file__).resolve().parent / "data" / "artifacts"))

        return RAGSettings(
            artifacts_root=artifacts_root,
            openrouter_api_key=_env_str("OPENROUTER_API_KEY"),
            openrouter_base_url=_env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            rerank_model=_env_str("RERANK_MODEL"),
            answer_model=_env_str("ANSWER_MODEL"),
            embedding_model=_env_str("EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_device=_env_str("EMBEDDING_DEVICE", "cpu"),
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
            top_k_sections=_env_int("TOP_K_SECTIONS", 8),
            top_k_chunks_raw=_env_int("TOP_K_CHUNKS_RAW", 120),
            max_chunks_after_filter=_env_int("MAX_CHUNKS_AFTER_FILTER", 24),
            max_pages=_env_int("MAX_PAGES", 25),
            neighbors=_env_int("RETRIEVER_NEIGHBORS", 0),
            restrict_to_primary_doc=_env_bool("RESTRICT_TO_PRIMARY_DOC", True),
            max_pages_for_rerank=_env_int("MAX_PAGES_FOR_RERANK", 12),
            max_pages_for_answer=_env_int("MAX_PAGES_FOR_ANSWER", 6),
            max_chars_per_page_for_answer=_env_int("MAX_CHARS_PER_PAGE_FOR_ANSWER", 12000),
            drop_empty_pages=_env_bool("DROP_EMPTY_PAGES", True),
        )


# -----------------------------------------------------------------------------
# HELPER: safe client construction
# -----------------------------------------------------------------------------

def _build_client(cls, **kwargs):
    """
    Instantiates a client class while only passing constructor args that it
    actually supports. This makes rag_main.py resilient to small differences
    between rerank/answer client constructors.
    """
    sig = inspect.signature(cls)
    accepted = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in kwargs:
            accepted[name] = kwargs[name]

    return cls(**accepted)


# -----------------------------------------------------------------------------
# RESULT MODELS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RAGPipelineResult:
    query: str
    kind: str
    answer: Any
    doc_id: str
    used_page_nos: List[int]
    retrieved_sections: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_pages: List[Dict[str, Any]]
    reranked_pages: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# SERVICE
# -----------------------------------------------------------------------------

class RAGService:
    def __init__(self, settings: Optional[RAGSettings] = None):
        self.settings = settings or RAGSettings.from_env()

        self.artifacts = ArtifactsStore(self.settings.artifacts_root)
        self.artifacts.validate()

        self.embedder = BgeM3Embedder(
            model_name=self.settings.embedding_model,
            device=self.settings.embedding_device,
            batch_size=self.settings.embedding_batch_size,
            expected_dim=self.artifacts.embedding_dim,
        )

        self.retriever = HierarchicalPageRetriever(
            artifacts=self.artifacts,
            embedder=self.embedder,
            top_k_sections=self.settings.top_k_sections,
            top_k_chunks_raw=self.settings.top_k_chunks_raw,
            max_chunks_after_filter=self.settings.max_chunks_after_filter,
            max_pages=self.settings.max_pages,
            neighbors=self.settings.neighbors,
            drop_empty_pages=self.settings.drop_empty_pages,
            restrict_to_primary_doc=self.settings.restrict_to_primary_doc,
        )

        rerank_client = _build_client(
            RerankModelClient,
            model_name=self.settings.rerank_model,
            model=self.settings.rerank_model,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
        )
        answer_client = _build_client(
            AnswerModelClient,
            model_name=self.settings.answer_model,
            model=self.settings.answer_model,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
        )

        self.reranker = PageReranker(client=rerank_client)
        self.answerer = RAGAnswerer(
            client=answer_client,
            max_chars_per_page=self.settings.max_chars_per_page_for_answer,
        )

    async def answer_query_full(self, query: str, kind: str = "text") -> RAGPipelineResult:
        query = (query or "").strip()
        if not query:
            raise ValueError("query must be a non-empty string")

        retrieval: RetrievalBundle = self.retriever.retrieve(query)

        if not retrieval.retrieved_pages:
            return RAGPipelineResult(
                query=query,
                kind=kind,
                answer="N/A",
                doc_id="",
                used_page_nos=[],
                retrieved_sections=[asdict(x) for x in retrieval.retrieved_sections],
                retrieved_chunks=[asdict(x) for x in retrieval.retrieved_chunks],
                retrieved_pages=[],
                reranked_pages=[],
            )

        pages_for_rerank = [
            {
                "doc_id": p.doc_id,
                "page_no": p.page_no,
                "text": p.text,
                "vector_score": p.vector_score,
            }
            for p in retrieval.retrieved_pages[: self.settings.max_pages_for_rerank]
        ]

        ranked_pages: List[RankedPage] = await self.reranker.rerank_pages(
            question=query,
            pages=pages_for_rerank,
        )

        ranked_lookup: Dict[int, RankedPage] = {rp.page_no: rp for rp in ranked_pages}

        reranked_page_dicts: List[Dict[str, Any]] = []
        for page in pages_for_rerank:
            rp = ranked_lookup.get(int(page["page_no"]))
            reranked_page_dicts.append(
                {
                    "doc_id": page["doc_id"],
                    "page_no": page["page_no"],
                    "text": page["text"],
                    "vector_score": page.get("vector_score", 0.0),
                    "llm_score": rp.llm_score if rp else 0.0,
                    "reasoning": rp.reasoning if rp else "",
                }
            )

        reranked_page_dicts.sort(key=lambda x: float(x.get("llm_score", 0.0)), reverse=True)

        pages_for_answer = reranked_page_dicts[: self.settings.max_pages_for_answer]

        answer_result: AnswerResult = await self.answerer.answer(
            question_text=query,
            kind=kind,
            pages=pages_for_answer,
        )

        return RAGPipelineResult(
            query=query,
            kind=kind,
            answer=answer_result.value,
            doc_id=answer_result.doc_id or (retrieval.primary_doc_id or ""),
            used_page_nos=answer_result.used_page_nos,
            retrieved_sections=[asdict(x) for x in retrieval.retrieved_sections],
            retrieved_chunks=[asdict(x) for x in retrieval.retrieved_chunks],
            retrieved_pages=[
                {
                    "doc_id": p.doc_id,
                    "page_no": p.page_no,
                    "text": p.text,
                    "vector_score": p.vector_score,
                }
                for p in retrieval.retrieved_pages
            ],
            reranked_pages=reranked_page_dicts,
        )

    async def answer_query(self, query: str, kind: str = "text") -> Any:
        result = await self.answer_query_full(query=query, kind=kind)
        return result.answer


# -----------------------------------------------------------------------------
# SINGLETON ACCESS
# -----------------------------------------------------------------------------

_service_singleton: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = RAGService()
    return _service_singleton


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

async def answer_query_full_async(query: str, kind: str = "text") -> Dict[str, Any]:
    service = get_rag_service()
    result = await service.answer_query_full(query=query, kind=kind)
    return asdict(result)


async def answer_query_async(query: str, kind: str = "text") -> Any:
    service = get_rag_service()
    return await service.answer_query(query=query, kind=kind)


def answer_query_full(query: str, kind: str = "text") -> Dict[str, Any]:
    return asyncio.run(answer_query_full_async(query=query, kind=kind))


def answer_query(query: str, kind: str = "text") -> Any:
    """
    Main integration entrypoint.

    Example:
        answer = answer_query("Что такое кислотно-основное титрование?")
    """
    return asyncio.run(answer_query_async(query=query, kind=kind))