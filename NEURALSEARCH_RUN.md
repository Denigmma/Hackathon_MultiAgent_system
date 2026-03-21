# Инструкция по запуску NeuralSearch

## 1. Подготовка окружения

```bash
cd Hackathon_MultiAgent_system-main
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Настройка OpenRouter

Создайте `.env` в корне проекта:

```env
# VseGPT / OpenAI-compatible API for NeuralSearch
VSEGPT_API_KEY=sk-or-vv-0eb5d58f7a04b7d2664726eaa2dc46df38195b086d78ce9826db414f7354e5f3
VSEGPT_MODEL=openai/gpt-5.4-nano-thinking-xhigh
```

## 3. Запуск

Разовый запрос:

```bash
python -m app.NeuralSearch.main "что такое RAG и как он работает"
```
