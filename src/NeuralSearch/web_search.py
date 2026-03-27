import logging
import os
from abc import ABC, abstractmethod
from urllib.parse import urlparse

import requests


logger = logging.getLogger(__name__)

DEFAULT_NUM_RESULTS = 10
SERPAPI_URL = "https://serpapi.com/search.json"


class SearchEngine(ABC):
    @abstractmethod
    def search(self, query: str, **kwargs) -> list[dict]:
        raise NotImplementedError


class SerpApiGoogleSearch(SearchEngine):
    def __init__(
        self,
        api_key: str | None = None,
        hl: str = "ru",
        gl: str = "ru",
        google_domain: str = "google.ru",
        timeout: int = 10,
    ):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        self.hl = hl
        self.gl = gl
        self.google_domain = google_domain
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError("SERPAPI_KEY is not set")

    def search(self, query: str, **kwargs) -> list[dict]:
        num_results = int(kwargs.get("num_results", DEFAULT_NUM_RESULTS))
        hl = kwargs.get("hl", self.hl)
        gl = kwargs.get("gl", self.gl)
        google_domain = kwargs.get("google_domain", self.google_domain)

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": num_results,
            "hl": hl,
            "gl": gl,
            "google_domain": google_domain,
        }

        logger.info(
            "SerpAPI search started | query=%s | num_results=%s | hl=%s | gl=%s | domain=%s | timeout=%ss",
            query,
            num_results,
            hl,
            gl,
            google_domain,
            self.timeout,
        )

        try:
            response = requests.get(SERPAPI_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            logger.warning(
                "SerpAPI search timed out and will be skipped | query=%s | timeout=%ss",
                query,
                self.timeout,
            )
            return []
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "SerpAPI search request failed and will be skipped | query=%s | error=%s",
                query,
                exc,
            )
            return []
        except ValueError as exc:
            logger.warning(
                "SerpAPI search returned invalid JSON and will be skipped | query=%s | error=%s",
                query,
                exc,
            )
            return []

        error_message = payload.get("error")
        if error_message:
            logger.warning(
                "SerpAPI search returned API error and will be skipped | query=%s | error=%s",
                query,
                error_message,
            )
            return []

        organic_results = payload.get("organic_results", [])
        parsed_results: list[dict] = []

        for index, item in enumerate(organic_results, start=1):
            url = (item.get("link") or "").strip()
            if not url:
                continue

            try:
                domain = urlparse(url).netloc.lower()
            except Exception:
                domain = ""

            parsed_results.append(
                {
                    "url": url,
                    "title": (item.get("title") or "").strip(),
                    "snippet": (item.get("snippet") or "").strip(),
                    "domain": domain,
                    "search_rank": int(item.get("position") or index),
                    "source": (item.get("source") or "").strip(),
                }
            )

        logger.info(
            "SerpAPI search finished | query=%s | organic_results=%s | parsed_results=%s",
            query,
            len(organic_results),
            len(parsed_results),
        )
        return parsed_results


class WebSearcher:
    def __init__(self, engine: SearchEngine):
        self.engine = engine

    def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS, **kwargs) -> list[dict]:
        try:
            results = self.engine.search(query, num_results=num_results, **kwargs)
        except Exception as exc:
            logger.warning(
                "Search engine failed and empty results will be returned | query=%s | error=%s",
                query,
                exc,
            )
            return []
        return results[:num_results] if results else []


def search_web(query: str, num_results: int = DEFAULT_NUM_RESULTS, **kwargs) -> list[dict]:
    if not isinstance(query, str) or not query.strip():
        logger.warning("Web search skipped because query is empty or invalid")
        return []

    try:
        searcher = WebSearcher(SerpApiGoogleSearch())
        results = searcher.search(query.strip(), num_results=num_results, **kwargs)
    except Exception as exc:
        logger.warning(
            "Web search initialization failed and empty results will be returned | query=%s | error=%s",
            query,
            exc,
        )
        return []

    logger.info(
        "Web search url collection finished | query_type=str | requested=%s | urls=%s",
        num_results,
        len(results),
    )
    return results


if __name__ == "__main__":
    query = "текущая цена нефти Brent USD за баррель"
    for item in search_web(query, num_results=5):
        print(item)