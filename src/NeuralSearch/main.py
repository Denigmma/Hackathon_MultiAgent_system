from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

from src.NeuralSearch.answer_generator import generate_answer
from src.NeuralSearch.paraphrase import ParaphaseMode, paraphrase_query
from src.NeuralSearch.reranker import rerank_documents
from src.NeuralSearch.web_search import search_web

Document = Tuple[str, str]


@dataclass
class NeuralSearchResult:
    user_query: str
    refined_queries: List[str]
    raw_documents: List[Document]
    reranked_documents: List[Document]
    answer: str


class NeuralSearchPipeline:
    def __init__(self, expand_queries: bool = False, search_results: int = 6, top_n: int = 5):
        self.expand_queries = expand_queries
        self.search_results = search_results
        self.top_n = top_n

    def run(self, user_query: str, history: Sequence[str] | None = None) -> NeuralSearchResult:
        mode = ParaphaseMode.EXPAND if self.expand_queries else ParaphaseMode.SIMPLIFY
        refined_queries = paraphrase_query(user_query, history=history, mode=mode)
        collected_docs: List[Document] = []
        seen_urls: set[str] = set()

        for refined in refined_queries:
            docs = search_web(refined, num_results=self.search_results)
            for url, content in docs:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                collected_docs.append((url, content))

        reranked = rerank_documents(user_query, collected_docs, top_n=self.top_n)
        answer = generate_answer(user_query, reranked, history=history)
        return NeuralSearchResult(
            user_query=user_query,
            refined_queries=refined_queries,
            raw_documents=collected_docs,
            reranked_documents=reranked,
            answer=answer,
        )


def run_query(user_query: str, history: Sequence[str] | None = None) -> str:
    return NeuralSearchPipeline().run(user_query, history=history).answer


def interactive_cli() -> None:
    print("NeuralSearch CLI. Для выхода: exit/quit/выход")
    history: List[str] = []
    pipeline = NeuralSearchPipeline()

    while True:
        try:
            query = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q", "выход"}:
            print("Выход.")
            break

        result = pipeline.run(query, history=history)
        history.append(f"Пользователь: {query}")
        history.append(f"AI-агент: {result.answer}")
        print("\nОтвет:\n")
        print(result.answer)


def main() -> str:
    parser = argparse.ArgumentParser(description="Run integrated NeuralSearch pipeline")
    parser.add_argument("query", nargs="*", help="text query for NeuralSearch")
    parser.add_argument("--expand", action="store_true", help="generate multiple paraphrases")
    parser.add_argument("--json", action="store_true", help="print full pipeline result as JSON")
    args = parser.parse_args()

    if not args.query:
        interactive_cli()

    query = " ".join(args.query).strip()
    pipeline = NeuralSearchPipeline(expand_queries=args.expand)
    result = pipeline.run(query)

    if args.json:
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)
    else:
        return result.answer


if __name__ == "__main__":
    raise SystemExit(main())
