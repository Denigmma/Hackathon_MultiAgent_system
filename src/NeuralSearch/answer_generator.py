from .models import llm
from langchain_core.messages import SystemMessage, HumanMessage

def generate_answer(query, documents: list[tuple[str, str]], history: list[str] | None = None) -> str:
    """
    Генерирует ответ на основе полученного из интернета контекста (documents) и пользовательского запроса (query).

    :param query: Пользовательский вопрос
    :param documents: Список строк с контекстом (результатами поиска), каждая строка включает ссылку
    :return: Ответ, основанный только на этих документах
    """

    prepared_documents: list[tuple[str, str]] = []
    for item in documents or []:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            prepared_documents.append((str(item[0]).strip(), str(item[1]).strip()))
        else:
            prepared_documents.append(("", str(item).strip()))

    context = (
        "\n\n".join(
            f"[{i + 1}] {url or 'источник без URL'}\n{doc}"
            for i, (url, doc) in enumerate(prepared_documents)
            if doc
        )
        if prepared_documents
        else "нет"
    )

    history_block = ""
    if history:
        history_lines = [str(item).strip() for item in history if str(item).strip()]
        if history_lines:
            history_block = "\n\nИстория диалога:\n" + "\n".join(history_lines[-6:])

    system_prompt = SystemMessage(
        content=(
            "Ты — помощник, который отвечает на вопросы пользователя исключительно на основе полученного контекста из интернета."
            " Не используй знания из своей памяти."
            " Каждое утверждение должно иметь ссылку на источник. "
            "Если информации недостаточно, честно сообщи об этом."
        )
    )

    human_prompt = HumanMessage(
        content=(
            f"Контекст из интернета:\n{context}\n\n"
            f"Вопрос пользователя: \"{query}\""
            f"{history_block}\n\n"
            "Ответь на этот вопрос, используя только указанный контекст и добавляя ссылки на источники."
        )
    )

    return llm.invoke([system_prompt, human_prompt]).strip()
