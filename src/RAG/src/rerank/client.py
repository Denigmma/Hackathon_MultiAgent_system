from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List

from dotenv import load_dotenv
from openai import AsyncOpenAI


@dataclass
class ModelClient:
    model_name: str
    base_url: str
    api_key: str
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def generate(self, payload: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        if isinstance(payload, dict):
            prompt = payload.get("prompt", "") or ""
            system_prompt = payload.get("system_prompt", "") or ""
        elif isinstance(payload, str):
            prompt = payload
            system_prompt = ""
        else:
            return {"error_flag": 1, "error_msg": f"Unsupported payload type: {type(payload)}", "text": None}

        if not isinstance(prompt, str):
            return {"error_flag": 1, "error_msg": "prompt must be str", "text": None}
        if not isinstance(system_prompt, str):
            return {"error_flag": 1, "error_msg": "system_prompt must be str", "text": None}

        messages: List[Dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=self.timeout_s,
            )
            text: Optional[str] = completion.choices[0].message.content if completion and completion.choices else ""
            return {"error_flag": 0, "error_msg": None, "text": text}
        except Exception as e:
            return {"error_flag": 1, "error_msg": str(e), "text": None}


def build_client_from_env(model_name: str, base_url_env: str = "OPENROUTER_BASE_URL", key_env: str = "OPENROUTER_API_KEY") -> ModelClient:
    load_dotenv()
    api_key = os.getenv(key_env)
    if not api_key:
        raise ValueError(f"Environment variable {key_env} is not set.")
    base_url = os.getenv(base_url_env, "https://openrouter.ai/api/v1")
    return ModelClient(model_name=model_name, base_url=base_url, api_key=api_key)
