import os
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
MODEL_EMPTY_RESPONSE_RETRIES = 3


class OpenRouterLLM:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        model_name = os.getenv("MODEL_WEB_SEARCH") or os.getenv("model_web_search")
        if not model_name:
            raise ValueError("MODEL_WEB_SEARCH (or model_web_search) is not set")

        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @staticmethod
    def _extract_role(message) -> str:
        role = getattr(message, "type", None)
        if role in {"system", "human", "ai", "assistant", "user"}:
            if role == "human":
                return "user"
            if role == "ai":
                return "assistant"
            return role

        if isinstance(message, dict):
            dict_role = message.get("role", "user")
            if dict_role == "human":
                return "user"
            if dict_role == "ai":
                return "assistant"
            return dict_role

        return "user"

    @staticmethod
    def _extract_content(message):
        content = getattr(message, "content", None)
        if content is not None:
            return content

        if isinstance(message, dict):
            return message.get("content", "")

        return str(message)

    def _normalize_messages(self, messages) -> list[dict]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        normalized = []
        for message in messages:
            normalized.append(
                {
                    "role": self._extract_role(message),
                    "content": self._extract_content(message),
                }
            )
        return normalized

    def invoke(self, messages) -> str:
        normalized_messages = self._normalize_messages(messages)
        for attempt in range(1, MODEL_EMPTY_RESPONSE_RETRIES + 1):
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=normalized_messages,
            )
            choices = getattr(completion, "choices", None) or []
            if not choices:
                if attempt < MODEL_EMPTY_RESPONSE_RETRIES:
                    time.sleep(1.0 * attempt)
                    continue
                return ""

            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            if message is None:
                if attempt < MODEL_EMPTY_RESPONSE_RETRIES:
                    time.sleep(1.0 * attempt)
                    continue
                return ""

            content = getattr(message, "content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(text)
                    else:
                        text = getattr(item, "text", None)
                        if text:
                            parts.append(text)
                content = "\n".join(parts)

            normalized_content = (content or "").strip()
            if normalized_content:
                return normalized_content

            if attempt < MODEL_EMPTY_RESPONSE_RETRIES:
                time.sleep(1.0 * attempt)

        return ""


class LazyResource:
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._instance: Any | None = None

    def _get_instance(self) -> Any:
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, item: str) -> Any:
        return getattr(self._get_instance(), item)


llm = LazyResource(OpenRouterLLM)
cross_encoder = LazyResource(lambda: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"))
bi_encoder = LazyResource(lambda: SentenceTransformer("intfloat/multilingual-e5-base"))
