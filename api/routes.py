from __future__ import annotations

import json
import logging
from queue import Empty

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatCreateResponse,
    ChatDetailResponse,
    ChatInfo,
    ChatListResponse,
    ChatMessageResponse,
    DeleteChatResponse,
    LoginRequest,
    MessageRequest,
    MessageResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)


def build_router(service) -> APIRouter:
    router = APIRouter()

    def format_sse(event_name: str, data: dict) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event_name}\ndata: {payload}\n\n"

    def require_user(user_id: str):
        user = service.get_user(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="Пользователь не найден.")
        return user

    def require_chat(user_id: str, chat_id: str):
        try:
            return service.get_chat(user_id, chat_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/session/login", response_model=UserResponse)
    def login(payload: LoginRequest) -> UserResponse:
        try:
            user = service.login(payload.username)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    @router.get("/api/users/{user_id}", response_model=UserResponse)
    def get_user(user_id: str) -> UserResponse:
        user = require_user(user_id)
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    @router.get("/api/users/{user_id}/chats", response_model=ChatListResponse)
    def list_chats(user_id: str) -> ChatListResponse:
        require_user(user_id)
        items = [
            ChatInfo(
                chat_id=chat.chat_id,
                title=chat.title,
                created_at=chat.created_at,
                updated_at=chat.updated_at,
            )
            for chat in service.list_chats(user_id)
        ]
        return ChatListResponse(items=items)

    @router.post("/api/users/{user_id}/chats", response_model=ChatCreateResponse)
    def create_chat(user_id: str) -> ChatCreateResponse:
        require_user(user_id)
        chat = service.create_chat(user_id)
        logger.info("Created chat %s for user %s", chat.chat_id, user_id)
        return ChatCreateResponse(
            chat_id=chat.chat_id,
            title=chat.title,
        )

    @router.delete("/api/users/{user_id}/chats/{chat_id}", response_model=DeleteChatResponse)
    def delete_chat(user_id: str, chat_id: str) -> DeleteChatResponse:
        require_user(user_id)
        if not service.delete_chat(user_id, chat_id):
            raise HTTPException(status_code=404, detail="Чат не найден.")
        logger.info("Deleted chat %s for user %s", chat_id, user_id)
        return DeleteChatResponse(success=True, chat_id=chat_id)

    @router.get("/api/users/{user_id}/chats/{chat_id}", response_model=ChatDetailResponse)
    def get_chat(user_id: str, chat_id: str) -> ChatDetailResponse:
        chat = require_chat(user_id, chat_id)

        messages = [
            ChatMessageResponse(
                message_id=message.message_id,
                role=message.role,
                content=message.content,
                agent=message.agent,
                kind=message.kind,
                created_at=message.created_at,
            )
            for message in service.list_messages(user_id, chat_id)
        ]
        return ChatDetailResponse(
            chat_id=chat.chat_id,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            messages=messages,
        )

    @router.get("/api/users/{user_id}/chats/{chat_id}/events")
    def stream_chat_events(user_id: str, chat_id: str):
        require_chat(user_id, chat_id)
        subscriber = service.event_hub.subscribe(chat_id)

        def event_stream():
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        event = subscriber.get(timeout=15)
                    except Empty:
                        yield ": keep-alive\n\n"
                        continue

                    yield format_sse(event["event"], event["data"])
            finally:
                service.event_hub.unsubscribe(chat_id, subscriber)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post("/api/users/{user_id}/chats/{chat_id}/messages", response_model=MessageResponse)
    def send_message(user_id: str, chat_id: str, payload: MessageRequest) -> MessageResponse:
        require_chat(user_id, chat_id)
        text = payload.message.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Сообщение пустое.")

        logger.info("Received message for user=%s chat=%s", user_id, chat_id)
        result = service.handle_message(user_id, chat_id, text)
        return MessageResponse(
            success=result.success,
            assistant_message=result.assistant_message,
            progress_count=result.progress_count,
            error=result.error,
        )

    return router

