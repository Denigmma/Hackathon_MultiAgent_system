from __future__ import annotations

import numpy as np
from typing import Iterable

from sklearn.metrics.pairwise import cosine_similarity

from .models import cross_encoder, bi_encoder

def preprocess(texts, is_query=True):
    prefix = "query: " if is_query else "passage: "
    return [prefix + text.strip() for text in texts]

def batch_encode(texts, is_query=True, batch_size=8):
    """Encodes texts using batching."""
    processed_texts = preprocess(texts, is_query)
    embeddings = []
    for i in range(0, len(processed_texts), batch_size):
        batch = processed_texts[i:i + batch_size]
        embeddings_batch = bi_encoder.encode(batch)
        embeddings.extend(embeddings_batch)
    return np.array(embeddings)


def _normalize_documents(documents: Iterable[tuple[str, str] | str]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []

    for item in documents:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            url = str(item[0]).strip()
            text = str(item[1]).strip()
        else:
            url = ""
            text = str(item).strip()

        if text:
            normalized.append((url, text))

    return normalized

def mmr(query_embedding, doc_embeddings, documents, top_n=5, lambda_param=0.7):
    """
    Maximal Marginal Relevance (MMR) for document selection.
    Selects documents that are both relevant to the query and diverse from each other.
    
    :param query_embedding: Embedding of the query.
    :param doc_embeddings: List of document embeddings.
    :param documents: List of documents.
    :param top_n: Number of documents to select.
    :param lambda_param: Trade-off parameter between relevance and diversity.
    :return: List of selected documents.
    """
    
    if len(doc_embeddings) == 0 or len(documents) == 0:
        return []

    selected_indices = []
    remaining_indices = list(range(len(doc_embeddings)))
    query_similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    max_to_select = min(int(top_n), len(remaining_indices))

    for _ in range(max_to_select):
        if not remaining_indices:
            break

        mmr_scores = []
        for i in remaining_indices:
            diversity_score = max(
                cosine_similarity([doc_embeddings[i]], [doc_embeddings[j]])[0][0]
                for j in selected_indices
            ) if selected_indices else 0
            
            mmr_score = lambda_param * query_similarities[i] - (1 - lambda_param) * diversity_score
            mmr_scores.append((i, mmr_score))

        if not mmr_scores:
            break

        best_doc = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_doc[0])
        remaining_indices.remove(best_doc[0])
    
    return [documents[idx] for idx in selected_indices]

def rerank_documents(query, documents, top_n=5, mmr_lambda=0.5, batch_size=8):
    if not isinstance(query, str) or not query.strip():
        return []

    normalized_documents = _normalize_documents(documents)
    if not normalized_documents:
        return []

    query_embedding = batch_encode([query.strip()], is_query=True, batch_size=1)
    doc_embeddings = batch_encode(
        [text for _, text in normalized_documents],
        is_query=False,
        batch_size=batch_size,
    )
    return mmr(query_embedding, doc_embeddings, normalized_documents, top_n, mmr_lambda)
