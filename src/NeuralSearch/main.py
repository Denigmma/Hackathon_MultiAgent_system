import logging

from .web_search import search_web
from .reranker import rerank_documents
from .answer_generator import generate_answer


FINAL_SOURCE_COUNT = 5
RAW_SEARCH_RESULTS = 10
logger = logging.getLogger(__name__)


def ai_overview_pipeline(user_query, history=None):
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string")

    normalized_query = user_query.strip()

    raw_urls = search_web(normalized_query, num_results=RAW_SEARCH_RESULTS)
    logger.info(
        "Web pipeline search finished | user_query=%s | raw_url_count=%s",
        normalized_query,
        len(raw_urls or []),
    )
    if not raw_urls:
        return "Не удалось найти релевантные источники в интернете."

    from .url_parcer import build_search_corpus

    parsed_docs = build_search_corpus(
        normalized_query,
        raw_urls,
        max_urls=FINAL_SOURCE_COUNT,
    )
    logger.info(
        "Web pipeline corpus built | user_query=%s | parsed_doc_count=%s",
        normalized_query,
        len(parsed_docs),
    )
    if not parsed_docs:
        return "Не удалось извлечь содержимое из найденных веб-источников."

    top_docs = rerank_documents(
        normalized_query,
        parsed_docs,
        top_n=FINAL_SOURCE_COUNT,
    )
    logger.info(
        "Web pipeline rerank finished | user_query=%s | top_doc_count=%s",
        normalized_query,
        len(top_docs),
    )
    if not top_docs:
        return "Не удалось отобрать релевантные веб-источники для ответа."

    return generate_answer(normalized_query, top_docs, history=history)


if __name__ == "__main__":
    history = []
    while True:
        print("\n\n====================\n")
        q = input("Введите ваш вопрос: ")
        res = ai_overview_pipeline(q, history)
        history.append("Пользователь: " + q)
        history.append("AI-агент: " + res)

        print("\n📝 Ответ:\n", res)
