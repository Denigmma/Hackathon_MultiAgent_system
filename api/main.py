from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from api.routes import build_router
from src.webapp.chat_store import ChatStore
from src.webapp.event_hub import ChatEventHub
from src.webapp.service import MASChatService

BASE_DIR = Path(__file__).resolve().parent.parent
WEBUI_DIR = BASE_DIR / "webui"
STATIC_DIR = WEBUI_DIR / "static"
HTML_TEMPLATES_DIR = WEBUI_DIR / "templates"
APP_DB_PATH = Path(os.getenv("WEBAPP_DB_PATH", str(BASE_DIR / "storage" / "chat_app.sqlite3")))
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(title="Chemistry MultiAgent Web")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(HTML_TEMPLATES_DIR))

    store = ChatStore(APP_DB_PATH)
    event_hub = ChatEventHub()
    service = MASChatService(store=store, event_hub=event_hub)

    @app.get("/")
    def chat_page(request: Request):
        return templates.TemplateResponse(
            request,
            "chat.html",
            {
                "page_title": "Мультиагентная система химика-органика",
            },
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(build_router(service))
    return app


app = create_app()
