from __future__ import annotations

from dataclasses import dataclass

from src.mas_runtime import MASProgressEvent, run_mas_query
from src.webapp.chat_store import ChatRecord, ChatStore, MessageRecord, UserRecord
from src.webapp.event_hub import ChatEventHub

MAX_CONTEXT_MESSAGES = 6

AGENT_LABELS = {
    "Supervisor": "Главный агент",
    "StructureAnalyzer": "Агент по анализу структуры",
    "SynthesisProtocolSearchAgent": "Агент по поиску методик синтеза",
    "LiteratureRAGAgent": "Агент по литературному поиску",
    "Assistant": "Ассистент",
    "System": "Система",
    "Пользователь": "Пользователь",
}


@dataclass(frozen=True)
class MessageExecutionResult:
    success: bool
    assistant_message: str
    progress_count: int
    error: str | None = None


class MASChatService:
    def __init__(self, store: ChatStore, event_hub: ChatEventHub) -> None:
        self.store = store
        self.event_hub = event_hub

    def login(self, username: str) -> UserRecord:
        return self.store.login_or_create_user(username)

    def get_user(self, user_id: str) -> UserRecord | None:
        return self.store.get_user(user_id)

    def create_chat(self, user_id: str) -> ChatRecord:
        return self.store.create_chat(user_id)

    def list_chats(self, user_id: str) -> list[ChatRecord]:
        return self.store.list_chats(user_id)

    def get_chat(self, user_id: str, chat_id: str) -> ChatRecord:
        chat = self.store.get_chat(user_id, chat_id)
        if chat is None:
            raise KeyError("Чат не найден.")
        return chat

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        return self.store.delete_chat(user_id, chat_id)

    def list_messages(self, user_id: str, chat_id: str) -> list[MessageRecord]:
        return self.store.list_messages(user_id, chat_id)

    def _build_contextual_query(self, user_id: str, chat_id: str, user_message: str) -> str:
        recent = self.store.list_recent_messages(
            user_id,
            chat_id,
            limit=MAX_CONTEXT_MESSAGES,
            roles=("user", "assistant"),
        )
        if not recent:
            return user_message

        lines = []
        for item in recent:
            role = "Пользователь" if item.role == "user" else "Ассистент"
            lines.append(f"{role}: {item.content}")

        history_block = "\n".join(lines).strip()
        if not history_block:
            return user_message

        return (
            "Ниже приведён краткий контекст предыдущего диалога. "
            "Используй его только если новый запрос ссылается на предыдущие сообщения.\n\n"
            f"{history_block}\n\n"
            f"Новый запрос пользователя:\n{user_message}"
        )

    @staticmethod
    def _display_agent_name(agent: str | None) -> str | None:
        if agent is None:
            return None
        return AGENT_LABELS.get(agent, agent)

    @staticmethod
    def _event_to_text(event: MASProgressEvent) -> str:
        agent = (event.agent or "").strip()
        content = (event.content or "").strip()
        if agent and agent != "Supervisor":
            return content or "Промежуточный шаг агента завершён."
        return content or "Система обрабатывает запрос."

    @staticmethod
    def _message_to_payload(message: MessageRecord) -> dict[str, Any]:
        return {
            "message_id": message.message_id,
            "role": message.role,
            "content": message.content,
            "agent": message.agent,
            "kind": message.kind,
            "created_at": message.created_at.isoformat(),
        }

    def handle_message(self, user_id: str, chat_id: str, user_message: str) -> MessageExecutionResult:
        self.get_chat(user_id, chat_id)
        contextual_query = self._build_contextual_query(user_id, chat_id, user_message)

        self.store.add_message(
            user_id,
            chat_id,
            "user",
            user_message,
            agent=self._display_agent_name("Пользователь"),
            kind="user_message",
        )
        self.store.update_title_if_default(user_id, chat_id, user_message)
        self.store.touch_chat(user_id, chat_id)

        progress_count = 0

        def publish_progress(event: MASProgressEvent) -> None:
            nonlocal progress_count
            content = self._event_to_text(event)
            message = self.store.add_message(
                user_id,
                chat_id,
                "progress",
                content,
                agent=self._display_agent_name(event.agent),
                kind=event.kind,
            )
            progress_count += 1
            self.event_hub.publish(
                chat_id,
                "progress",
                self._message_to_payload(message),
            )

        try:
            result = run_mas_query(contextual_query, on_event=publish_progress)
            answer = result.answer.strip() or "Система завершила обработку, но не вернула текстовый ответ."
            assistant_message = self.store.add_message(
                user_id,
                chat_id,
                "assistant",
                answer,
                agent=self._display_agent_name("Assistant"),
                kind="final_answer",
            )
            self.store.touch_chat(user_id, chat_id)
            self.event_hub.publish(
                chat_id,
                "completed",
                self._message_to_payload(assistant_message),
            )
            return MessageExecutionResult(
                success=True,
                assistant_message=answer,
                progress_count=progress_count,
            )
        except Exception as exc:
            error_text = f"Ошибка при обработке запроса: {exc}"
            error_message = self.store.add_message(
                user_id,
                chat_id,
                "assistant",
                error_text,
                agent=self._display_agent_name("System"),
                kind="error",
            )
            self.store.touch_chat(user_id, chat_id)
            self.event_hub.publish(
                chat_id,
                "error",
                self._message_to_payload(error_message),
            )
            return MessageExecutionResult(
                success=False,
                assistant_message=error_text,
                progress_count=progress_count,
                error=error_text,
            )
