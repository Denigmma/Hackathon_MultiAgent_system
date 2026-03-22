from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    manifest: Path
    merged_reports_dir: Path
    sectioned_reports_dir: Path
    chunked_reports_dir: Path
    vector_dbs_dir: Path

    @staticmethod
    def from_root(root: Path) -> "ArtifactPaths":
        root = root.resolve()
        return ArtifactPaths(
            root=root,
            manifest=root / "manifest.json",
            merged_reports_dir=root / "merged_reports",
            sectioned_reports_dir=root / "sectioned_reports",
            chunked_reports_dir=root / "chunked_reports",
            vector_dbs_dir=root / "vector_dbs",
        )


class _LRUCache:
    def __init__(self, max_items: int = 8):
        self.max_items = int(max_items)
        self._data: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


@dataclass(frozen=True)
class SectionMeta:
    index_pos: int
    doc_id: str
    section_id: str
    title: str
    start_page: int
    end_page: int
    summary: str
    is_fallback_window: bool = False


@dataclass(frozen=True)
class ChunkMeta:
    index_pos: int
    chunk_id_num: int
    chunk_id: str
    doc_id: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    length_tokens: int
    chunk_type: str = "content"


class ArtifactsStore:
    """
    Loads and serves artifacts from the new hierarchical archive layout:

      - manifest.json
      - merged_reports/{doc_id}.json
      - sectioned_reports/{doc_id}.json
      - chunked_reports/{doc_id}.json
      - vector_dbs/sections.faiss + sections.meta.json
      - vector_dbs/chunks.faiss + chunks.meta.json

    All loads are lazy and cached (LRU).
    """

    def __init__(
        self,
        artifacts_root: str | Path,
        cache_merged: int = 4,
        cache_sectioned: int = 4,
        cache_chunked: int = 4,
        cache_index: int = 4,
        cache_meta: int = 8,
    ):
        self.paths = ArtifactPaths.from_root(Path(artifacts_root))

        if not self.paths.root.exists():
            raise FileNotFoundError(f"Artifacts root not found: {self.paths.root}")
        if not self.paths.manifest.exists():
            raise FileNotFoundError(f"manifest.json not found: {self.paths.manifest}")
        if not self.paths.merged_reports_dir.exists():
            raise FileNotFoundError(f"merged_reports dir not found: {self.paths.merged_reports_dir}")
        if not self.paths.vector_dbs_dir.exists():
            raise FileNotFoundError(f"vector_dbs dir not found: {self.paths.vector_dbs_dir}")

        self.manifest: Dict[str, Any] = _read_json(self.paths.manifest)
        self.page_base: int = int(self.manifest.get("page_base", 1))

        emb = self.manifest.get("embeddings", {}) or {}
        self.embedding_dim: Optional[int] = emb.get("dim", None)
        if self.embedding_dim is not None:
            self.embedding_dim = int(self.embedding_dim)

        docs = self.manifest.get("documents")
        if not isinstance(docs, dict) or not docs:
            raise ValueError("manifest.documents is missing or invalid")
        self.documents: Dict[str, Any] = docs

        self.indices_info: Dict[str, Any] = self.manifest.get("indices", {}) or {}
        self.sections_info: Dict[str, Any] = self.manifest.get("sections", {}) or {}

        self._merged_cache = _LRUCache(max_items=cache_merged)
        self._sectioned_cache = _LRUCache(max_items=cache_sectioned)
        self._chunked_cache = _LRUCache(max_items=cache_chunked)
        self._index_cache = _LRUCache(max_items=cache_index)
        self._meta_cache = _LRUCache(max_items=cache_meta)

        self._section_meta_by_id: Optional[Dict[str, SectionMeta]] = None
        self._chunk_meta_by_id: Optional[Dict[str, ChunkMeta]] = None
        self._sections_by_doc: Optional[Dict[str, List[SectionMeta]]] = None

    # -------------------------------------------------------------------------
    # Document-level access
    # -------------------------------------------------------------------------

    def list_doc_ids(self) -> List[str]:
        return sorted(self.documents.keys())

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self.documents

    def get_document_meta(self, doc_id: str) -> Dict[str, Any]:
        doc = self.documents.get(doc_id)
        return doc if isinstance(doc, dict) else {}

    def get_doc_title(self, doc_id: str) -> str:
        doc = self.get_document_meta(doc_id)
        title = doc.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()

        meta = doc.get("meta", {})
        if isinstance(meta, dict):
            alt = meta.get("title")
            if isinstance(alt, str) and alt.strip():
                return alt.strip()

        return doc_id

    def get_doc_stats(self, doc_id: str) -> Dict[str, Any]:
        doc = self.get_document_meta(doc_id)
        stats = doc.get("stats", {})
        return stats if isinstance(stats, dict) else {}

    # -------------------------------------------------------------------------
    # Report loading
    # -------------------------------------------------------------------------

    def load_merged_report(self, doc_id: str) -> Dict[str, Any]:
        key = f"merged:{doc_id}"
        cached = self._merged_cache.get(key)
        if cached is not None:
            return cached

        path = self.paths.merged_reports_dir / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"merged_report not found: {path}")

        report = _read_json(path)
        if not isinstance(report, dict) or "content" not in report:
            raise ValueError(f"Invalid merged_report schema: {path}")

        self._merged_cache.set(key, report)
        return report

    def load_sectioned_report(self, doc_id: str) -> Dict[str, Any]:
        key = f"sectioned:{doc_id}"
        cached = self._sectioned_cache.get(key)
        if cached is not None:
            return cached

        path = self.paths.sectioned_reports_dir / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"sectioned_report not found: {path}")

        report = _read_json(path)
        if not isinstance(report, dict) or "content" not in report:
            raise ValueError(f"Invalid sectioned_report schema: {path}")

        self._sectioned_cache.set(key, report)
        return report

    def load_chunked_report(self, doc_id: str) -> Dict[str, Any]:
        key = f"chunked:{doc_id}"
        cached = self._chunked_cache.get(key)
        if cached is not None:
            return cached

        path = self.paths.chunked_reports_dir / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"chunked_report not found: {path}")

        report = _read_json(path)
        if not isinstance(report, dict) or "content" not in report:
            raise ValueError(f"Invalid chunked_report schema: {path}")

        self._chunked_cache.set(key, report)
        return report

    # -------------------------------------------------------------------------
    # Page access
    # -------------------------------------------------------------------------

    def get_pages(self, doc_id: str) -> List[Dict[str, Any]]:
        merged = self.load_merged_report(doc_id)
        pages = merged.get("content", {}).get("pages", [])
        return pages if isinstance(pages, list) else []

    def get_page(self, doc_id: str, page_no: int) -> Optional[Dict[str, Any]]:
        for page in self.get_pages(doc_id):
            if isinstance(page, dict) and int(page.get("page_no", -1)) == int(page_no):
                return page
        return None

    def get_page_text(self, doc_id: str, page_no: int) -> str:
        page = self.get_page(doc_id, page_no)
        if not isinstance(page, dict):
            return ""
        text = page.get("text", "")
        return text if isinstance(text, str) else ""

    def get_page_range_texts(self, doc_id: str, start_page: int, end_page: int) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        for page_no in range(int(start_page), int(end_page) + 1):
            text = self.get_page_text(doc_id, page_no)
            if text:
                out.append((page_no, text))
        return out

    def page_no_to_page_index(self, page_no: int) -> int:
        return int(page_no) - 1

    # -------------------------------------------------------------------------
    # Global FAISS access
    # -------------------------------------------------------------------------

    def load_sections_faiss_index(self):
        return self._load_faiss_index_by_name("sections")

    def load_chunks_faiss_index(self):
        return self._load_faiss_index_by_name("chunks")

    def _load_faiss_index_by_name(self, name: str):
        key = f"faiss:{name}"
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached

        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FAISS is not installed. Install a compatible faiss-cpu package."
            ) from e

        path = self.paths.vector_dbs_dir / f"{name}.faiss"
        if not path.exists():
            raise FileNotFoundError(f"faiss index not found: {path}")

        index = faiss.read_index(str(path))

        if self.embedding_dim is not None and int(index.d) != int(self.embedding_dim):
            raise ValueError(
                f"FAISS dim mismatch for {name}: "
                f"index.d={index.d} vs manifest.embeddings.dim={self.embedding_dim}"
            )

        self._index_cache.set(key, index)
        return index

    # -------------------------------------------------------------------------
    # Meta loading
    # -------------------------------------------------------------------------

    def load_sections_meta(self) -> List[SectionMeta]:
        key = "meta:sections"
        cached = self._meta_cache.get(key)
        if cached is not None:
            return cached

        path = self.paths.vector_dbs_dir / "sections.meta.json"
        if not path.exists():
            raise FileNotFoundError(f"sections.meta.json not found: {path}")

        raw = _read_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"Invalid sections.meta.json schema: {path}")

        meta: List[SectionMeta] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid section meta item at position {i}")

            meta.append(
                SectionMeta(
                    index_pos=int(item.get("index_pos", i)),
                    doc_id=str(item["doc_id"]),
                    section_id=str(item["section_id"]),
                    title=str(item.get("title", "")),
                    start_page=int(item["start_page"]),
                    end_page=int(item["end_page"]),
                    summary=str(item.get("summary", "")),
                    is_fallback_window=bool(item.get("is_fallback_window", False)),
                )
            )

        meta.sort(key=lambda x: x.index_pos)
        self._meta_cache.set(key, meta)
        return meta

    def load_chunks_meta(self) -> List[ChunkMeta]:
        key = "meta:chunks"
        cached = self._meta_cache.get(key)
        if cached is not None:
            return cached

        path = self.paths.vector_dbs_dir / "chunks.meta.json"
        if not path.exists():
            raise FileNotFoundError(f"chunks.meta.json not found: {path}")

        raw = _read_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"Invalid chunks.meta.json schema: {path}")

        meta: List[ChunkMeta] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid chunk meta item at position {i}")

            meta.append(
                ChunkMeta(
                    index_pos=int(item.get("index_pos", i)),
                    chunk_id_num=int(item.get("chunk_id_num", i)),
                    chunk_id=str(item["chunk_id"]),
                    doc_id=str(item["doc_id"]),
                    section_id=str(item["section_id"]),
                    section_title=str(item.get("section_title", "")),
                    page_start=int(item["page_start"]),
                    page_end=int(item["page_end"]),
                    length_tokens=int(item.get("length_tokens", 0)),
                    chunk_type=str(item.get("type", "content")),
                )
            )

        meta.sort(key=lambda x: x.index_pos)
        self._meta_cache.set(key, meta)
        return meta

    # -------------------------------------------------------------------------
    # Meta lookup helpers
    # -------------------------------------------------------------------------

    def get_section_meta_by_index_pos(self, index_pos: int) -> SectionMeta:
        meta = self.load_sections_meta()
        if index_pos < 0 or index_pos >= len(meta):
            raise IndexError(f"Section index_pos out of range: {index_pos}")
        item = meta[index_pos]
        if item.index_pos != index_pos:
            # fallback if meta order is not strictly identical to index_pos
            return self.get_section_meta(item.section_id)
        return item

    def get_chunk_meta_by_index_pos(self, index_pos: int) -> ChunkMeta:
        meta = self.load_chunks_meta()
        if index_pos < 0 or index_pos >= len(meta):
            raise IndexError(f"Chunk index_pos out of range: {index_pos}")
        item = meta[index_pos]
        if item.index_pos != index_pos:
            return self.get_chunk_meta(item.chunk_id)
        return item

    def get_section_meta(self, section_id: str) -> SectionMeta:
        if self._section_meta_by_id is None:
            self._section_meta_by_id = {m.section_id: m for m in self.load_sections_meta()}
        try:
            return self._section_meta_by_id[section_id]
        except KeyError as e:
            raise KeyError(f"Unknown section_id: {section_id}") from e

    def get_chunk_meta(self, chunk_id: str) -> ChunkMeta:
        if self._chunk_meta_by_id is None:
            self._chunk_meta_by_id = {m.chunk_id: m for m in self.load_chunks_meta()}
        try:
            return self._chunk_meta_by_id[chunk_id]
        except KeyError as e:
            raise KeyError(f"Unknown chunk_id: {chunk_id}") from e

    def get_sections_by_doc(self, doc_id: str) -> List[SectionMeta]:
        if self._sections_by_doc is None:
            grouped: Dict[str, List[SectionMeta]] = {}
            for item in self.load_sections_meta():
                grouped.setdefault(item.doc_id, []).append(item)
            self._sections_by_doc = grouped
        return list(self._sections_by_doc.get(doc_id, []))

    # -------------------------------------------------------------------------
    # Section/chunk access from reports
    # -------------------------------------------------------------------------

    def get_sections_from_report(self, doc_id: str) -> List[Dict[str, Any]]:
        report = self.load_sectioned_report(doc_id)
        sections = report.get("content", {}).get("sections", [])
        return sections if isinstance(sections, list) else []

    def get_chunks_from_report(self, doc_id: str) -> List[Dict[str, Any]]:
        report = self.load_chunked_report(doc_id)
        chunks = report.get("content", {}).get("chunks", [])
        return chunks if isinstance(chunks, list) else []

    def get_section_text(self, doc_id: str, section_id: str) -> str:
        for section in self.get_sections_from_report(doc_id):
            if isinstance(section, dict) and str(section.get("section_id")) == section_id:
                text = section.get("text", "")
                return text if isinstance(text, str) else ""
        return ""

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> None:
        if self.page_base != 1:
            raise ValueError(f"Expected page_base=1, got {self.page_base}")

        doc_ids = self.list_doc_ids()
        if not doc_ids:
            raise ValueError("No documents found in manifest")

        sections_index = self.load_sections_faiss_index()
        chunks_index = self.load_chunks_faiss_index()

        sections_meta = self.load_sections_meta()
        chunks_meta = self.load_chunks_meta()

        if self.embedding_dim is not None:
            if int(sections_index.d) != int(self.embedding_dim):
                raise ValueError(
                    "Embedding dim mismatch for sections index: "
                    f"{sections_index.d} vs {self.embedding_dim}"
                )
            if int(chunks_index.d) != int(self.embedding_dim):
                raise ValueError(
                    "Embedding dim mismatch for chunks index: "
                    f"{chunks_index.d} vs {self.embedding_dim}"
                )

        if int(sections_index.ntotal) != len(sections_meta):
            raise ValueError(
                f"sections index/meta size mismatch: ntotal={sections_index.ntotal}, meta={len(sections_meta)}"
            )
        if int(chunks_index.ntotal) != len(chunks_meta):
            raise ValueError(
                f"chunks index/meta size mismatch: ntotal={chunks_index.ntotal}, meta={len(chunks_meta)}"
            )

        sample_doc_id = doc_ids[0]
        merged = self.load_merged_report(sample_doc_id)
        pages = merged.get("content", {}).get("pages", [])
        if not isinstance(pages, list) or not pages:
            raise ValueError(f"Merged report pages missing for doc_id={sample_doc_id}")

        first_page_no = pages[0].get("page_no")
        if first_page_no != 1:
            raise ValueError(f"Expected first page_no=1, got {first_page_no}")

        sample_section = sections_meta[0]
        if not self.has_doc(sample_section.doc_id):
            raise ValueError(
                f"Section meta references unknown doc_id: {sample_section.doc_id}"
            )
        if sample_section.start_page > sample_section.end_page:
            raise ValueError(
                f"Invalid section page span: {sample_section.section_id} "
                f"{sample_section.start_page}>{sample_section.end_page}"
            )

        sample_chunk = chunks_meta[0]
        if not self.has_doc(sample_chunk.doc_id):
            raise ValueError(
                f"Chunk meta references unknown doc_id: {sample_chunk.doc_id}"
            )
        if sample_chunk.page_start > sample_chunk.page_end:
            raise ValueError(
                f"Invalid chunk page span: {sample_chunk.chunk_id} "
                f"{sample_chunk.page_start}>{sample_chunk.page_end}"
            )
        _ = self.get_section_meta(sample_chunk.section_id)