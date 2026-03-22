from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    artifacts_dir: str = "data/artifacts"
    questions_path: str = "data/questions.json"
    outputs_dir: str = "outputs"

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    rerank_model: str = "xiaomi/mimo-v2-flash:free"
    answer_model: str = "xiaomi/mimo-v2-flash:free"

    embedder_model: str = "BAAI/bge-m3"
    embedder_device: str = "cpu"
    embedder_batch_size: int = 64

    top_k_chunks: int = 80
    max_pages: int = 30
    neighbors: int = 0
    drop_empty_pages: bool = True

    max_pages_for_rerank: int = 18

    rerank_top_n_pages: int = 4
    reference_min_pages: int = 1
    reference_max_pages: int = 3

    max_concurrency_questions: int = 20

    # if you want to blend vector + llm rerank scores in final ranking:
    blend_vector_llm: bool = False
    blend_a: float = 0.3
    blend_b: float = 0.7

    @staticmethod
    def load() -> "AppConfig":
        load_dotenv()

        def _get_int(name: str, default: int) -> int:
            v = os.getenv(name)
            if v is None or v == "":
                return default
            return int(v)

        def _get_float(name: str, default: float) -> float:
            v = os.getenv(name)
            if v is None or v == "":
                return default
            return float(v)

        def _get_bool(name: str, default: bool) -> bool:
            v = os.getenv(name)
            if v is None or v == "":
                return default
            t = v.strip().lower()
            if t in {"1", "true", "yes", "y", "on"}:
                return True
            if t in {"0", "false", "no", "n", "off"}:
                return False
            return default

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment (.env).")

        return AppConfig(
            artifacts_dir=os.getenv("ARTIFACTS_DIR", "data/artifacts"),
            questions_path=os.getenv("QUESTIONS_PATH", "data/questions.json"),
            outputs_dir=os.getenv("OUTPUTS_DIR", "outputs"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openrouter_api_key=api_key,
            rerank_model=os.getenv("RERANK_MODEL", "openai/o4-mini"),
            answer_model=os.getenv("ANSWER_MODEL", "openai/o4-mini"),
            embedder_model=os.getenv("EMBEDDER_MODEL", "BAAI/bge-m3"),
            embedder_device=os.getenv("EMBEDDER_DEVICE", "cpu"),
            embedder_batch_size=_get_int("EMBEDDER_BATCH_SIZE", 64),
            top_k_chunks=_get_int("TOP_K_CHUNKS", 40),
            max_pages=_get_int("MAX_PAGES", 25),
            neighbors=_get_int("NEIGHBORS", 0),
            drop_empty_pages=_get_bool("DROP_EMPTY_PAGES", True),
            rerank_top_n_pages=_get_int("RERANK_TOP_N_PAGES", 6),
            reference_min_pages=_get_int("REFERENCE_MIN_PAGES", 1),
            reference_max_pages=_get_int("REFERENCE_MAX_PAGES", 3),
            max_concurrency_questions=_get_int("MAX_CONCURRENCY_QUESTIONS", 20),
            blend_vector_llm=_get_bool("BLEND_VECTOR_LLM", False),
            blend_a=_get_float("BLEND_A", 0.3),
            blend_b=_get_float("BLEND_B", 0.7),
        )
