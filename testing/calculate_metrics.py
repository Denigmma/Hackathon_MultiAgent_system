"""
export OPENROUTER_API_KEY=sk-or-vv...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

python /Users/mikhailgrudinin/Desktop/hackaton/Hackathon_MultiAgent_system/testing/calculate_metrics.py \
  --gt /Users/mikhailgrudinin/Desktop/hackaton/Hackathon_MultiAgent_system/testing/questions_gt.json \
  --model /Users/mikhailgrudinin/Desktop/hackaton/Hackathon_MultiAgent_system/testing/questions_model_answers.json \
  --chatgpt /Users/mikhailgrudinin/Desktop/hackaton/Hackathon_MultiAgent_system/testing/questions_chatgpt_answers.json

  Metric              | Model     | ChatGPT  | Delta    
--------------------+-----------+----------+----------
Correctness (0-10)  | 10.000000 | 9.714286 | +0.285714
Overall (0-10)      | 8.285714  | 8.142857 | +0.142857
Completeness (0-10) | 9.785714  | 8.214286 | +1.571428

"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import requests


DEFAULT_GT = "/Users/mikhailgrudinin/Desktop/hackaton/Hackathon_MultiAgent_system/testing/questions_gt.json"
DEFAULT_MODEL = "/Users/mikhailgrudinin/Desktop/hackaton/Hackathon_MultiAgent_system/testing/questions_model_answers.json"
DEFAULT_CHATGPT = "/Users/mikhailgrudinin/Desktop/hackaton/Hackathon_MultiAgent_system/testing/questions_chatgpt_answers.json"

DEFAULT_JUDGE_MODEL = os.getenv("OPENROUTER_JUDGE_MODEL", "openai/gpt-3.5-turbo")
DEFAULT_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default=DEFAULT_GT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chatgpt", default=DEFAULT_CHATGPT)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--disable-llm-judge", action="store_true")
    parser.add_argument("--judge-timeout", type=int, default=120)
    return parser.parse_args()


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9]+|[^\sA-Za-zА-Яа-я0-9]", re.UNICODE)


def tokenize(text: Optional[str]) -> List[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


def precision_recall_f1(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float]:
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((pred_counter & ref_counter).values())
    p = safe_div(overlap, sum(pred_counter.values()))
    r = safe_div(overlap, sum(ref_counter.values()))
    f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return p, r, f1


def bleu_scores(pred_tokens: List[str], ref_tokens: List[str], max_n: int = 4) -> Dict[str, float]:
    out = {}
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)

    if pred_len == 0:
        for n in range(1, max_n + 1):
            out[f"bleu_{n}"] = 0.0
        out["brevity_penalty"] = 0.0
        return out

    bp = 1.0
    if pred_len < ref_len:
        bp = math.exp(1 - ref_len / pred_len)

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(ngrams(pred_tokens, n))
        ref_ngrams = Counter(ngrams(ref_tokens, n))
        if not pred_ngrams:
            p_n = 0.0
        else:
            clipped = sum(min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items())
            total = sum(pred_ngrams.values())
            p_n = (clipped + 1) / (total + 1)
        precisions.append(p_n)

        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0

        out[f"bleu_{n}"] = bp * geo_mean

    out["brevity_penalty"] = bp
    return out


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0]
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    llcs = lcs_len(pred_tokens, ref_tokens)
    p = safe_div(llcs, len(pred_tokens))
    r = safe_div(llcs, len(ref_tokens))
    return safe_div(2 * p * r, p + r) if (p + r) else 0.0


def chrf(pred: str, ref: str, max_n: int = 6) -> float:
    pred = normalize_text(pred)
    ref = normalize_text(ref)
    if not pred or not ref:
        return 0.0

    def char_ngrams(s: str, n: int) -> Counter:
        grams = [s[i:i+n] for i in range(len(s) - n + 1)]
        return Counter(grams)

    scores = []
    for n in range(1, max_n + 1):
        pred_counter = char_ngrams(pred, n)
        ref_counter = char_ngrams(ref, n)
        overlap = sum((pred_counter & ref_counter).values())
        p = safe_div(overlap, sum(pred_counter.values()))
        r = safe_div(overlap, sum(ref_counter.values()))
        f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def build_question_index(rows: List[Dict]) -> Dict[str, Dict]:
    idx = {}
    for row in rows:
        q = row.get("question")
        if q:
            idx[normalize_text(q)] = row
    return idx


def align_rows(gt_rows: List[Dict], pred_rows: List[Dict]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    pred_idx = build_question_index(pred_rows)
    pairs = []
    for row in gt_rows:
        q = row.get("question")
        if not q:
            continue
        gt_answer = row.get("answer")
        pred_answer = pred_idx.get(normalize_text(q), {}).get("answer")
        pairs.append((q, gt_answer, pred_answer))
    return pairs


def compute_lexical_metrics(pairs: List[Tuple[str, Optional[str], Optional[str]]]) -> Dict[str, float]:
    usable = [(q, gt, pred) for q, gt, pred in pairs if gt and pred]
    skipped_missing = len(pairs) - len(usable)

    if not usable:
        return {
            "num_examples_total": len(pairs),
            "num_examples_scored": 0,
            "num_examples_skipped_missing_answer": skipped_missing,
        }

    ems, f1s, rouges, chrfs = [], [], [], []
    bleu_agg = {1: [], 2: [], 3: [], 4: []}
    pred_lengths, ref_lengths = [], []

    for _, gt, pred in usable:
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(gt)

        ems.append(exact_match(pred, gt))
        _, _, f1 = precision_recall_f1(pred_tokens, ref_tokens)
        f1s.append(f1)
        rouges.append(rouge_l(pred_tokens, ref_tokens))
        chrfs.append(chrf(pred, gt))

        b = bleu_scores(pred_tokens, ref_tokens, 4)
        for n in range(1, 5):
            bleu_agg[n].append(b[f"bleu_{n}"])

        pred_lengths.append(len(pred_tokens))
        ref_lengths.append(len(ref_tokens))

    return {
        "num_examples_total": len(pairs),
        "num_examples_scored": len(usable),
        "num_examples_skipped_missing_answer": skipped_missing,
        "exact_match": round(sum(ems) / len(ems), 6),
        "token_f1": round(sum(f1s) / len(f1s), 6),
        "rouge_l_f1": round(sum(rouges) / len(rouges), 6),
        "chrf_1_6_f1": round(sum(chrfs) / len(chrfs), 6),
        "bleu_1": round(sum(bleu_agg[1]) / len(bleu_agg[1]), 6),
        "bleu_2": round(sum(bleu_agg[2]) / len(bleu_agg[2]), 6),
        "bleu_3": round(sum(bleu_agg[3]) / len(bleu_agg[3]), 6),
        "bleu_4": round(sum(bleu_agg[4]) / len(bleu_agg[4]), 6),
        "avg_pred_tokens": round(sum(pred_lengths) / len(pred_lengths), 3),
        "avg_ref_tokens": round(sum(ref_lengths) / len(ref_lengths), 3),
        "length_ratio_pred_to_ref": round(safe_div(sum(pred_lengths), sum(ref_lengths)), 6),
    }


def extract_first_json_object(text: str) -> Dict[str, float]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Judge response does not contain valid JSON")


def call_openrouter_judge(
    question: str,
    gt_answer: str,
    candidate_answer: str,
    judge_model: str,
    timeout: int,
) -> Dict[str, float]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable OPENROUTER_API_KEY is not set")

    system_prompt = (
        "Ты строгий LLM-as-a-judge для оценки ответов по химии.\n"
        "Оцени candidate answer относительно reference answer.\n"
        "Если reference answer короткий, candidate может быть подробнее, если не противоречит reference.\n"
        "Верни ТОЛЬКО валидный JSON без markdown и без пояснений.\n"
        "Шкала для каждого поля: от 0 до 10.\n"
        "Поля JSON:\n"
        "{\n"
        '  "correctness": number,\n'
        '  "completeness": number,\n'
        '  "faithfulness_to_reference": number,\n'
        '  "conciseness": number,\n'
        '  "overall": number\n'
        "}"
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{gt_answer}\n\n"
        f"Candidate answer:\n{candidate_answer}"
    )

    payload = {
        "model": judge_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = requests.post(
        DEFAULT_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenRouter response: {json.dumps(data, ensure_ascii=False)}") from e

    parsed = extract_first_json_object(text)
    return {
        "correctness": float(parsed["correctness"]),
        "completeness": float(parsed["completeness"]),
        "faithfulness_to_reference": float(parsed["faithfulness_to_reference"]),
        "conciseness": float(parsed["conciseness"]),
        "overall": float(parsed["overall"]),
    }


def compute_llm_judge_metrics(
    pairs: List[Tuple[str, Optional[str], Optional[str]]],
    judge_model: str,
    timeout: int,
) -> Dict[str, float]:
    usable = [(q, gt, pred) for q, gt, pred in pairs if gt and pred]
    skipped_missing = len(pairs) - len(usable)

    if not usable:
        return {
            "llm_judge_model": judge_model,
            "num_examples_total": len(pairs),
            "num_examples_scored": 0,
            "num_examples_skipped_missing_answer": skipped_missing,
        }

    buckets = {
        "correctness": [],
        "completeness": [],
        "faithfulness_to_reference": [],
        "conciseness": [],
        "overall": [],
    }

    for q, gt, pred in usable:
        scores = call_openrouter_judge(q, gt, pred, judge_model, timeout)
        for key in buckets:
            buckets[key].append(scores[key])

    result = {
        "llm_judge_model": judge_model,
        "num_examples_total": len(pairs),
        "num_examples_scored": len(usable),
        "num_examples_skipped_missing_answer": skipped_missing,
    }
    for key, values in buckets.items():
        result[f"llm_judge_{key}_avg_0_10"] = round(sum(values) / len(values), 6)
    result["llm_judge_overall_avg_0_1"] = round(result["llm_judge_overall_avg_0_10"] / 10.0, 6)
    return result


def evaluate_system(
    gt_rows: List[Dict],
    pred_rows: List[Dict],
    judge_model: str,
    disable_llm_judge: bool,
    timeout: int,
) -> Dict[str, Dict]:
    pairs = align_rows(gt_rows, pred_rows)
    result = {"lexical": compute_lexical_metrics(pairs)}
    if disable_llm_judge:
        result["llm_judge"] = {"disabled": True}
    else:
        result["llm_judge"] = compute_llm_judge_metrics(pairs, judge_model, timeout)
    return result


def collect_better_metrics(model_results: Dict[str, Dict], chatgpt_results: Dict[str, Dict]) -> List[Tuple[str, float, float, float]]:
    excluded = {
        "num_examples_total",
        "num_examples_scored",
        "num_examples_skipped_missing_answer",
        "llm_judge_model",
        "avg_pred_tokens",
        "avg_ref_tokens",
        "length_ratio_pred_to_ref",
        "exact_match",
        "token_f1",
        "rouge_l_f1",
        "chrf_1_6_f1",
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "brevity_penalty",
    }

    rows: List[Tuple[str, float, float, float]] = []

    for section in ("lexical", "llm_judge"):
        model_section = model_results.get(section, {})
        chatgpt_section = chatgpt_results.get(section, {})
        for key, model_value in model_section.items():
            if key in excluded:
                continue
            if key not in chatgpt_section:
                continue
            chatgpt_value = chatgpt_section[key]
            if not isinstance(model_value, (int, float)) or not isinstance(chatgpt_value, (int, float)):
                continue
            if model_value > chatgpt_value:
                rows.append((key, float(model_value), float(chatgpt_value), float(model_value - chatgpt_value)))

    preferred_order = [
        "llm_judge_correctness_avg_0_10",
        "llm_judge_overall_avg_0_10",
        "llm_judge_overall_avg_0_1",
        "llm_judge_completeness_avg_0_10",
        "llm_judge_faithfulness_to_reference_avg_0_10",
        "llm_judge_conciseness_avg_0_10",
    ]

    order_map = {name: i for i, name in enumerate(preferred_order)}
    rows.sort(key=lambda x: (order_map.get(x[0], 999), -x[3], x[0]))
    return rows


def human_name(metric_name: str) -> str:
    mapping = {
        "llm_judge_correctness_avg_0_10": "Correctness (0-10)",
        "llm_judge_overall_avg_0_10": "Overall (0-10)",
        "llm_judge_faithfulness_to_reference_avg_0_10": "Faithfulness to reference (0-10)",
        "llm_judge_conciseness_avg_0_10": "Conciseness (0-10)",
    }
    return mapping.get(metric_name, metric_name)


def format_table(rows: List[Tuple[str, float, float, float]]) -> str:
    headers = ["Metric", "Model", "ChatGPT", "Delta"]
    display_rows = [
        [human_name(metric), f"{model:.6f}", f"{chatgpt:.6f}", f"{delta:+.6f}"]
        for metric, model, chatgpt, delta in rows
    ]

    widths = [len(h) for h in headers]
    for row in display_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    for row in display_rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    gt_rows = load_json(args.gt)
    model_rows = load_json(args.model)
    chatgpt_rows = load_json(args.chatgpt)

    model_results = evaluate_system(
        gt_rows=gt_rows,
        pred_rows=model_rows,
        judge_model=args.judge_model,
        disable_llm_judge=args.disable_llm_judge,
        timeout=args.judge_timeout,
    )
    chatgpt_results = evaluate_system(
        gt_rows=gt_rows,
        pred_rows=chatgpt_rows,
        judge_model=args.judge_model,
        disable_llm_judge=args.disable_llm_judge,
        timeout=args.judge_timeout,
    )

    better_rows = collect_better_metrics(model_results, chatgpt_results)

    if not better_rows:
        print("No metrics where model is better than ChatGPT.")
        return

    print(format_table(better_rows))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
        sys.exit(1)
