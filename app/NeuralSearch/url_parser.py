from __future__ import annotations

import re
from html import unescape
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)


def parse_url(url: str, timeout: int = 15) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        allow_redirects=True,
    )
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "img", "header", "footer"]):
        tag.decompose()

    body = soup.body or soup
    text = body.get_text("\n", strip=True)
    text = unescape(text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text


def split_text(text: str, chunk_size: int = 1400, overlap: int = 250) -> List[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(clean):
        chunk = clean[start : start + chunk_size]
        chunks.append(chunk)
        start += step
    return chunks


def extract_relevant(
    query: str,
    text: str,
    max_document_length: int = 4500,
    chunk_size: int = 1400,
) -> str:
    chunks = split_text(text, chunk_size=chunk_size)
    if not chunks:
        return ""

    corpus = [query] + chunks
    vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform(corpus)
    query_vec = matrix[0:1]
    chunk_vecs = matrix[1:]
    scores = cosine_similarity(query_vec, chunk_vecs).flatten()

    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    selected: List[str] = []
    total_len = 0
    for chunk in ranked_chunks:
        if total_len + len(chunk) > max_document_length:
            break
        selected.append(chunk)
        total_len += len(chunk)

    return "\n\n".join(selected)
