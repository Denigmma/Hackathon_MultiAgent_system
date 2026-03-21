"""
Документация к модулю VseGPTWrapper
===================================

Этот модуль предоставляет удобную оболочку над OpenAI-совместимым API (настроенным по умолчанию на VseGPT).
Он поддерживает ведение истории диалога, потоковую генерацию, работу с JSON, а также продвинутый двухпроходный режим с самопроверкой (draft -> self-check -> final).

Конфигурация (VseGPTConfig)
---------------------------
Для инициализации клиента требуется объект конфигурации `VseGPTConfig`:
* api_key: str — Ваш API ключ.
* base_url: str — URL API (по умолчанию "https://api.vsegpt.ru/v1").
* model: str — Имя модели (по умолчанию "anthropic/claude-3-haiku").
* temperature: float — Креативность (по умолчанию 0.7).
* max_tokens: int — Максимальное количество токенов (по умолчанию 3000).
* n: int — Количество вариантов ответа (по умолчанию 1).
* default_headers: Dict[str, str] — Заголовки по умолчанию (по умолчанию {"X-Title": "My App"}).

Инициализация
-------------
config = VseGPTConfig(api_key="ВАШ_КЛЮЧ")
llm = VseGPTWrapper(config)

Универсальный метод (Рекомендуемый)
-----------------------------------
Самый простой способ взаимодействия — использовать метод `ask`. Он автоматически маршрутизирует запрос под капотом в зависимости от флагов.

* ask(prompt, system_prompt=None, json_mode=False, use_history=False, self_check=False, **kwargs)
  - prompt: Текст запроса.
  - json_mode=True: Возвращает распарсенный JSON (или dict с данными самопроверки).
  - use_history=True: Учитывает и обновляет историю диалога.
  - self_check=True: Включает двухпроходный режим (модель генерирует черновик, затем проверяет его на ошибки и возвращает исправленный вариант).

Базовые методы генерации
------------------------
Если вам нужен более тонкий контроль, используйте специализированные методы:

* complete_text(prompt, system_prompt=None, ...) -> str
  Одиночный запрос. Возвращает только текст ответа.

* complete_json(prompt, system_prompt=None, ...) -> Any
  Одиночный запрос, который требует от модели JSON. Возвращает распарсенный JSON (словари/списки).

* chat(user_prompt, system_prompt=None, messages=None, use_history=False, save_to_history=False, ...) -> Dict
  Основной метод отправки сообщений. Возвращает словарь:
  {"content": str, "reasoning": str|None, "raw": openai_response_object}

* chat_with_history(user_prompt, system_prompt=None, ...) -> Dict
  Аналог `chat`, но автоматически читает и сохраняет сообщения в `self.history`.

* stream(user_prompt, system_prompt=None, ...) -> Generator[str]
  Потоковая генерация ответа. Идеально для длинных текстов, выдает ответ по частям (yield).

Продвинутые методы (Самопроверка / Reasoning)
---------------------------------------------
Режим самопроверки заставляет модель сначала написать черновик, а затем выступить в роли "строгого рецензента", найдя и исправив свои же ошибки.

* answer_with_self_check(prompt, system_prompt=None, ...) -> Dict
  Возвращает словарь с этапами генерации:
  {"draft": str, "check": Dict (результат проверки), "content": str (финальный ответ)}

* answer_with_self_check_json(prompt, system_prompt=None, ...) -> Dict
  То же самое, но гарантирует, что финальный ответ парсится как JSON.
  Возвращает: {"draft": str, "check": Dict, "raw_text": str, "data": Any}

Управление историей и состоянием
--------------------------------
* reset_history() — Полностью очищает историю текущего диалога.
* add_message(role, content) — Ручное добавление сообщения в историю (role обычно "user", "assistant" или "system").
* set_model(model) — Быстрое изменение модели "на лету" без пересоздания конфига.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Generator, Tuple

from openai import OpenAI

Message = Dict[str, str]


@dataclass
class VseGPTConfig:
    api_key: str
    base_url: str = "https://api.vsegpt.ru/v1"
    model: str = "anthropic/claude-3-haiku"
    temperature: float = 0.7
    max_tokens: int = 3000
    n: int = 1
    default_headers: Dict[str, str] = field(
        default_factory=lambda: {"X-Title": "My App"}
    )


class VseGPTWrapper:
    """
    Оболочка для работы с VseGPT через OpenAI-compatible API.

    Добавлено:
    - двухпроходный режим: draft -> self-check -> final
    - методы для самопроверки ответа
    - безопасный "reasoning mode" без раскрытия скрытого хода мыслей
    """

    def __init__(self, config: VseGPTConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.history: List[Message] = []

    def set_model(self, model: str) -> None:
        self.config.model = model

    def reset_history(self) -> None:
        self.history = []

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _merge_messages(
        self,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None,
        use_history: bool = False,
    ) -> List[Message]:
        merged: List[Message] = []

        if use_history:
            merged.extend(self.history)

        if system_prompt:
            merged.insert(0, {"role": "system", "content": system_prompt})

        if messages:
            merged.extend(messages)

        if user_prompt is not None:
            merged.append({"role": "user", "content": user_prompt})

        return merged

    def _extract_text(self, response: Any) -> str:
        return response.choices[0].message.content or ""

    def _extract_reasoning(self, response: Any) -> Optional[str]:
        msg = response.choices[0].message
        return getattr(msg, "reasoning", None)

    def _call_chat_completion(
        self,
        merged_messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Any:
        return self.client.chat.completions.create(
            model=model or self.config.model,
            messages=merged_messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            n=self.config.n if n is None else n,
            extra_headers=extra_headers or self.config.default_headers,
            **kwargs,
        )  # type: ignore

    def _build_internal_reasoning_system_prompt(
        self, system_prompt: Optional[str] = None
    ) -> str:
        base = (
            "Ты решаешь задачу внутри, но не показываешь ход рассуждений. "
            "Сначала выведи только итоговый ответ, без внутреннего анализа, без черновиков и без скрытых шагов."
        )
        if system_prompt:
            return f"{system_prompt}\n\n{base}"
        return base

    def _build_self_check_prompt(self, draft_answer: str) -> str:
        return (
            "Проверь следующий ответ на ошибки, противоречия, пропуски и неточности. "
            "Верни строго JSON со следующими полями:\n"
            "{\n"
            '  "ok": boolean,\n'
            '  "issues": [string, ...],\n'
            '  "corrected_answer": string\n'
            "}\n\n"
            f"Ответ для проверки:\n{draft_answer}"
        )

    def _parse_json_safely(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        use_history: bool = False,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        save_to_history: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Обычный запрос к модели.
        Возвращает словарь:
        {
            "content": str,
            "reasoning": Optional[str],
            "raw": response
        }
        """
        merged_messages = self._merge_messages(
            user_prompt=user_prompt,
            messages=messages,
            system_prompt=system_prompt,
            use_history=use_history,
        )

        response = self._call_chat_completion(
            merged_messages=merged_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            extra_headers=extra_headers,
            **kwargs,
        )

        content = self._extract_text(response)
        reasoning = self._extract_reasoning(response)

        if save_to_history:
            if system_prompt and not use_history and not self.history:
                self.history.append({"role": "system", "content": system_prompt})
            elif (
                system_prompt
                and not use_history
                and self.history
                and self.history[0]["role"] != "system"
            ):
                self.history.insert(0, {"role": "system", "content": system_prompt})

            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": content})

        return {
            "content": content,
            "reasoning": reasoning,
            "raw": response,
        }

    def chat_with_history(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Диалоговый режим: использует self.history.
        После ответа автоматически дописывает user/assistant в историю.
        """
        return self.chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            use_history=True,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            extra_headers=extra_headers,
            save_to_history=True,
            **kwargs,
        )

    def chat_messages(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Запрос, когда сообщения уже собраны вручную.
        """
        merged_messages = self._merge_messages(
            messages=messages,
            system_prompt=system_prompt,
            use_history=False,
        )

        response = self._call_chat_completion(
            merged_messages=merged_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            extra_headers=extra_headers,
            **kwargs,
        )

        return {
            "content": self._extract_text(response),
            "reasoning": self._extract_reasoning(response),
            "raw": response,
        }

    def stream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        use_history: bool = False,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Потоковая генерация текста.
        """
        merged_messages = self._merge_messages(
            user_prompt=user_prompt,
            messages=messages,
            system_prompt=system_prompt,
            use_history=use_history,
        )

        stream = self.client.chat.completions.create(
            model=model or self.config.model,
            messages=merged_messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            n=self.config.n if n is None else n,
            extra_headers=extra_headers or self.config.default_headers,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def json_response(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        use_history: bool = False,
        model: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Запрос с попыткой получить JSON.
        Возвращает:
        {
            "data": dict | list | None,
            "raw_text": str,
            "raw": response
        }
        """
        merged_messages = self._merge_messages(
            user_prompt=user_prompt,
            messages=messages,
            system_prompt=system_prompt,
            use_history=use_history,
        )

        request_kwargs = dict(
            model=model or self.config.model,
            messages=merged_messages,
            temperature=temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            extra_headers=extra_headers or self.config.default_headers,
            **kwargs,
        )

        if response_format is not None:
            request_kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**request_kwargs)

        text = self._extract_text(response)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None

        return {
            "data": data,
            "raw_text": text,
            "raw": response,
        }

    def complete_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """
        Самый простой вариант: prompt -> text.
        """
        result = self.chat(
            user_prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            **kwargs,
        )
        return result["content"]

    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Any:
        """
        Самый простой JSON-вариант: prompt -> parsed JSON.
        """
        result = self.json_response(
            user_prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            response_format={"type": "json_output"},
            **kwargs,
        )
        return result["data"]

    # -------------------------
    # Самопроверка / reasoning
    # -------------------------

    def draft_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = False,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """
        Первый проход: делает внутренний черновик ответа.
        Ход рассуждений наружу не выводится.
        """
        reasoning_system_prompt = self._build_internal_reasoning_system_prompt(
            system_prompt
        )
        result = self.chat(
            user_prompt=prompt,
            system_prompt=reasoning_system_prompt,
            use_history=use_history,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            **kwargs,
        )
        return result["content"]

    def self_check_text(
        self,
        draft_answer: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Второй проход: проверяет черновик и возвращает структурированный результат.
        """
        check_prompt = self._build_self_check_prompt(draft_answer)

        check_system_prompt = (
            "Ты строгий рецензент. Проверяй фактические ошибки, логические несостыковки, "
            "неполноту, двусмысленность и формат. Верни только валидный JSON."
        )
        if system_prompt:
            check_system_prompt = f"{system_prompt}\n\n{check_system_prompt}"

        result = self.json_response(
            user_prompt=check_prompt,
            system_prompt=check_system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            response_format={"type": "json_output"},
            **kwargs,
        )

        data = result["data"]
        if isinstance(data, dict):
            return data

        return {
            "ok": False,
            "issues": ["Не удалось распарсить JSON самопроверки."],
            "corrected_answer": draft_answer,
            "raw_text": result["raw_text"],
        }

    def answer_with_self_check(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = False,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        self_check_temperature: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Двухпроходный режим:
        1) draft
        2) self-check
        3) финальный ответ = corrected_answer, если он есть, иначе draft
        """
        draft = self.draft_text(
            prompt=prompt,
            system_prompt=system_prompt,
            use_history=use_history,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            **kwargs,
        )

        check = self.self_check_text(
            draft_answer=draft,
            system_prompt=system_prompt,
            model=model,
            temperature=self_check_temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            **kwargs,
        )

        final_answer = draft
        if isinstance(check, dict):
            corrected = check.get("corrected_answer")
            if isinstance(corrected, str) and corrected.strip():
                final_answer = corrected.strip()

        return {
            "draft": draft,
            "check": check,
            "content": final_answer,
        }

    def answer_with_self_check_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = False,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        self_check_temperature: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Аналог answer_with_self_check, но для задач, где финальный ответ нужен в JSON.
        """
        draft = self.draft_text(
            prompt=prompt,
            system_prompt=system_prompt,
            use_history=use_history,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            **kwargs,
        )

        check = self.self_check_text(
            draft_answer=draft,
            system_prompt=system_prompt,
            model=model,
            temperature=self_check_temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            **kwargs,
        )

        final_text = draft
        if isinstance(check, dict):
            corrected = check.get("corrected_answer")
            if isinstance(corrected, str) and corrected.strip():
                final_text = corrected.strip()

        try:
            final_data = json.loads(final_text)
        except json.JSONDecodeError:
            final_data = None

        return {
            "draft": draft,
            "check": check,
            "raw_text": final_text,
            "data": final_data,
        }

    def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        use_history: bool = False,
        self_check: bool = False,
        **kwargs,
    ) -> Any:
        """
        Универсальный метод:
        - json_mode=False -> строка
        - json_mode=True  -> распарсенный JSON

        Дополнительно:
        - self_check=True включает двухпроходную самопроверку.
        """
        if json_mode:
            if self_check:
                return self.answer_with_self_check_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    use_history=use_history,
                    **kwargs,
                )

            if use_history:
                return self.json_response(
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    use_history=True,
                    response_format={"type": "json_output"},
                    **kwargs,
                )

            return self.complete_json(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )

        if self_check:
            return self.answer_with_self_check(
                prompt=prompt,
                system_prompt=system_prompt,
                use_history=use_history,
                **kwargs,
            )

        if use_history:
            return self.chat_with_history(
                user_prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )

        return self.complete_text(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )
