from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from src.NeuralSearch.url_parser import extract_relevant, parse_url

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
DEFAULT_NUM_RESULTS = 5


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str = ""
    snippet: str = ""


class DuckDuckGoSearch:
    search_url = "https://html.duckduckgo.com/html/"

    def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> List[SearchHit]:
        response = requests.post(
            self.search_url,
            data={"q": query},
            timeout=20,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results: List[SearchHit] = []
        for block in soup.select(".result"):
            link = block.select_one(".result__title a")
            snippet = block.select_one(".result__snippet")
            if not link or not link.get("href"):
                continue
            results.append(
                SearchHit(
                    url=link.get("href"),
                    title=link.get_text(" ", strip=True),
                    snippet=snippet.get_text(" ", strip=True) if snippet else "",
                )
            )
            if len(results) >= num_results:
                break
        return results


@dataclass(frozen=True)
class RetrievedDocument:
    url: str
    title: str
    snippet: str
    content: str

    def as_tuple(self) -> Tuple[str, str]:
        return self.url, self.content


def _fetch_document(query: str, hit: SearchHit) -> RetrievedDocument | None:
    try:
        parsed = parse_url(hit.url)
        relevant = extract_relevant(query, parsed)
        if not relevant.strip():
            relevant = hit.snippet
        return RetrievedDocument(
            url=hit.url,
            title=hit.title,
            snippet=hit.snippet,
            content=relevant.strip(),
        )
    except Exception:
        if hit.snippet:
            return RetrievedDocument(
                url=hit.url,
                title=hit.title,
                snippet=hit.snippet,
                content=hit.snippet,
            )
        return None


def search_web(query: str, num_results: int = DEFAULT_NUM_RESULTS) -> List[Tuple[str, str]]:
    engine = DuckDuckGoSearch()
    hits = engine.search(query, num_results=num_results)
    if not hits:
        return []

    documents: List[RetrievedDocument] = []
    with ThreadPoolExecutor(max_workers=min(6, len(hits))) as executor:
        futures = [executor.submit(_fetch_document, query, hit) for hit in hits]
        for future in as_completed(futures):
            doc = future.result()
            if doc is not None:
                documents.append(doc)

    ordered = sorted(
        documents,
        key=lambda doc: next((idx for idx, hit in enumerate(hits) if hit.url == doc.url), 10**6),
    )
    return [doc.as_tuple() for doc in ordered]
