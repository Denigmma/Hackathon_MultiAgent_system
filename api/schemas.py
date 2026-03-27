from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)


class UserResponse(BaseModel):
    user_id: str
    username: str
    created_at: datetime
    updated_at: datetime


class ChatCreateResponse(BaseModel):
    chat_id: str
    title: str


class ChatInfo(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatListResponse(BaseModel):
    items: list[ChatInfo]


class ChatMessageResponse(BaseModel):
    message_id: int
    role: str
    content: str
    agent: Optional[str] = None
    kind: Optional[str] = None
    created_at: datetime


class ChatDetailResponse(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageResponse]


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


class MessageResponse(BaseModel):
    success: bool
    assistant_message: str
    progress_count: int
    error: Optional[str] = None


class DeleteChatResponse(BaseModel):
    success: bool
    chat_id: str

