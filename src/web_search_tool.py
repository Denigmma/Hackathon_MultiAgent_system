from __future__ import annotations

import logging
from typing import Optional

from src.NeuralSearch.main import ai_overview_pipeline


logger = logging.getLogger(__name__)


class WebSearchToolError(Exception):
    pass


class WebSearchTool:
    def search(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise WebSearchToolError("query must be a non-empty string.")

        normalized_query = query.strip()
        logger.info("Web search started | query=%s", normalized_query)
        try:
            result = ai_overview_pipeline(normalized_query)
        except Exception as exc:
            logger.exception("Web search failed | query=%s", normalized_query)
            raise WebSearchToolError(f"Web search failed: {exc}") from exc

        if result is None:
            logger.info("Web search finished | query=%s | empty result", normalized_query)
            return ""

        if isinstance(result, str):
            normalized = result.strip()
            logger.info(
                "Web search finished | query=%s | response=%s",
                normalized_query,
                normalized[:300],
            )
            return normalized

        normalized = str(result).strip()
        logger.info(
            "Web search finished | query=%s | response=%s",
            normalized_query,
            normalized[:300],
        )
        return normalized


_default_web_search_tool: Optional[WebSearchTool] = None


def init_web_search_tool() -> WebSearchTool:
    global _default_web_search_tool
    _default_web_search_tool = WebSearchTool()
    return _default_web_search_tool


def get_web_search_tool() -> WebSearchTool:
    global _default_web_search_tool
    if _default_web_search_tool is None:
        _default_web_search_tool = WebSearchTool()
    return _default_web_search_tool


def search_web(query: str) -> str:
    return get_web_search_tool().search(query)
