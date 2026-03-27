import logging
import requests

from urllib.parse import urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md 

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.text_splitter import Language
except ModuleNotFoundError:  # pragma: no cover
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_text_splitters import Language

from .models import cross_encoder


logger = logging.getLogger(__name__)


def _normalize_search_result(item) -> dict | None:
    if isinstance(item, str):
        normalized_url = item.strip()
        if not normalized_url:
            return None
        return {
            "url": normalized_url,
            "title": "",
            "snippet": "",
            "score": 0.0,
            "search_rank": 10**9,
        }

    if isinstance(item, dict):
        normalized_url = str(item.get("url") or "").strip()
        if not normalized_url:
            return None
        return {
            "url": normalized_url,
            "title": str(item.get("title") or "").strip(),
            "snippet": str(item.get("snippet") or "").strip(),
            "score": float(item.get("score") or 0.0),
            "search_rank": int(item.get("search_rank") or 0),
        }

    return None


def _build_candidate_text(item: dict) -> str:
    parts = [
        item.get("title", ""),
        item.get("snippet", ""),
        item.get("url", ""),
    ]
    return "\n".join(part for part in parts if part).strip()


def _rank_search_results(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return []

    candidate_texts = [_build_candidate_text(item) for item in results]
    pairs = [(query, text) for text in candidate_texts]
    scores = cross_encoder.predict(pairs)

    ranked_results = []
    for item, score in zip(results, scores):
        enriched = dict(item)
        enriched["score"] = float(score)
        ranked_results.append(enriched)

    ranked_results.sort(
        key=lambda item: (item.get("score", 0.0), -item.get("search_rank", 10**9)),
        reverse=True,
    )
    return ranked_results

def parse_url(url: str) -> str:
    """
    Parses a URL and returns the HTML content in markdown format.
    
    :param url (str): The URL to parse.
    :return str: A markdown representation of the HTML content.
    """
    
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {url}")

    response = requests.get(
        url,
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0 AgentDocSystem/1.0"},
    )
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
    
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')

    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    main_node = soup.find("main") or soup.find("article") or soup.body
    if main_node is None:
        raise ValueError(f"No parseable HTML body found for {url}")

    markdown_content = md(str(main_node))
    return markdown_content

def extract_relevant(query: str, text: str, min_per_chunk: int = 1024, max_document_length: int=7500) -> str:
    """
    Extracts relevant information from the text using 8-layered BERT
    
    :param query (str): The query to search for in the text.
    :param text (str): The text to extract information from.
    :param min_per_chunk (int): The minimum number of characters per chunk.
    :param max_document_length (int): The maximum length of the document.
    
    :return str: The extracted information.
    """
    
    splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=min_per_chunk,
    chunk_overlap=min_per_chunk//2
    )
    
    chunks = splitter.split_text(text)
    
    def batched_predict(pairs, batch_size=8):
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            results.extend(cross_encoder.predict(batch))
        return results
    
    pairs = [(query, chunk) for chunk in chunks]
    scores = batched_predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    relevant_chunks = []
    length = 0
    for score, chunk in ranked:
        if length + len(chunk) > max_document_length:
            break
        relevant_chunks.append(chunk)
        length += len(chunk)
    return "\n\n".join(relevant_chunks)


def build_search_corpus(query: str, urls: list[str | dict], max_urls: int = 5) -> list[tuple[str, str]]:
    corpus: list[tuple[str, str]] = []

    if not isinstance(query, str) or not query.strip():
        return corpus

    logger.info(
        "Build search corpus started | query=%s | input_urls=%s | max_urls=%s",
        query.strip(),
        len(urls or []),
        max_urls,
    )

    seen_urls: set[str] = set()
    failed_domains: set[str] = set()
    normalized_results = []
    for item in urls:
        normalized_item = _normalize_search_result(item)
        if normalized_item is None:
            continue
        normalized_results.append(normalized_item)

    normalized_results.sort(key=lambda item: item.get("search_rank", 10**9))
    logger.info(
        "Search results prepared for parsing | query=%s | candidates=%s | top_candidates=%s",
        query.strip(),
        len(normalized_results),
        [
            {
                "url": item["url"],
                "search_rank": item.get("search_rank", 10**9),
                "title": item["title"][:80],
            }
            for item in normalized_results[:10]
        ],
    )

    for item in normalized_results:
        normalized_url = item["url"]
        title = item["title"]
        snippet = item["snippet"]
        result_score = item.get("score", 0.0)

        if not normalized_url or normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)
        domain = urlparse(normalized_url).netloc.lower()
        if domain and domain in failed_domains:
            logger.info("Skipping URL because domain already failed in this request | domain=%s | url=%s", domain, normalized_url)
            continue

        logger.info(
            "Parsing URL | url=%s | score=%.2f | title=%s | snippet=%s",
            normalized_url,
            result_score,
            title[:120],
            snippet[:160],
        )

        try:
            parsed_content = parse_url(normalized_url)
            relevant_text = extract_relevant(query.strip(), parsed_content)
        except Exception as exc:
            if domain:
                failed_domains.add(domain)
            logger.warning("URL parsing failed | url=%s | error=%s", normalized_url, exc)
            continue

        if relevant_text.strip():
            corpus.append((normalized_url, relevant_text.strip()))
            logger.info(
                "URL added to corpus | url=%s | corpus_size=%s",
                normalized_url,
                len(corpus),
            )
        else:
            logger.info("URL parsed but no relevant text found | url=%s", normalized_url)

        if len(corpus) >= max_urls:
            break

    logger.info("Build search corpus finished | final_corpus_size=%s", len(corpus))
    return corpus

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    query = "what is an Artificial Intelligence"
    
    try:
        parsed_content = parse_url(url)
        relevant_text = extract_relevant(query, parsed_content)
        print(relevant_text)
    except Exception as e:
        print(f"Error: {e}")
