from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Document = Tuple[str, str]


def preprocess(texts: Sequence[str], is_query: bool = True) -> List[str]:
    prefix = "query: " if is_query else "passage: "
    return [prefix + (text or "").strip() for text in texts]


def batch_encode(texts: Sequence[str], is_query: bool = True) -> np.ndarray:
    processed = preprocess(texts, is_query=is_query)
    if not processed:
        return np.empty((0, 0))
    return np.array(processed, dtype=object)


def _vectorize_for_ranking(query: str, documents: Sequence[Document]) -> tuple[np.ndarray, np.ndarray]:
    passages = [doc_text for _, doc_text in documents]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    matrix = vectorizer.fit_transform([query] + passages)
    query_embedding = matrix[0:1]
    doc_embeddings = matrix[1:]
    return query_embedding, doc_embeddings


def mmr(
    query_embedding,
    doc_embeddings,
    documents: Sequence[Document],
    top_n: int = 5,
    lambda_param: float = 0.7,
) -> List[Document]:
    if not documents:
        return []

    top_n = min(top_n, len(documents))
    selected_indices: List[int] = []
    remaining_indices = list(range(len(documents)))
    query_similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()

    for _ in range(top_n):
        best_idx = None
        best_score = None
        for i in remaining_indices:
            diversity_score = 0.0
            if selected_indices:
                diversity_score = max(
                    cosine_similarity(doc_embeddings[i], doc_embeddings[j])[0][0] for j in selected_indices
                )
            score = lambda_param * float(query_similarities[i]) - (1 - lambda_param) * float(diversity_score)
            if best_score is None or score > best_score:
                best_idx = i
                best_score = score
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [documents[idx] for idx in selected_indices]


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_n: int = 5,
    mmr_lambda: float = 0.7,
) -> List[Document]:
    if not documents:
        return []
    query_embedding, doc_embeddings = _vectorize_for_ranking(query, documents)
    return mmr(query_embedding, doc_embeddings, documents, top_n=top_n, lambda_param=mmr_lambda)
