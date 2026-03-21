from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from app.llm_client import VseGPTConfig, VseGPTWrapper


@dataclass(frozen=True)
class NeuralSearchConfig:
    api_key_env: str = "VSEGPT_API_KEY"
    model_env: str = "VSEGPT_MODEL"
    default_model: str = "anthropic/claude-3-haiku"
    base_url: str = "https://api.vsegpt.ru/v1"
    temperature: float = 0.2
    max_tokens: int = 3000
    app_title: str = "Hackathon MultiAgent NeuralSearch"


def _load_project_env() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break


@lru_cache(maxsize=1)
def get_llm() -> VseGPTWrapper:
    _load_project_env()
    cfg = NeuralSearchConfig()
    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Не найден API ключ VseGPT. Установите переменную окружения {cfg.api_key_env} или добавьте её в .env в корне проекта."
        )

    model = os.getenv(cfg.model_env, cfg.default_model)
    return VseGPTWrapper(
        VseGPTConfig(
            api_key=api_key,
            base_url=cfg.base_url,
            model=model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            default_headers={"X-Title": cfg.app_title},
        )
    )
