# llm_compression/baseline.py
#
# Baseline: answer MedQA-style MCQs directly from the original question text (NO compression),
# then compare against gold labels loaded from a separate answers JSONL.

import os
import json
import re
from typing import Dict, Any, Iterable, List
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI


# -------------------------
# ENV + OpenAI client
# -------------------------
ROOT_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=ROOT_ENV, override=True)
client = OpenAI()

# -------------------------
# Local input files
# -------------------------
THIS_DIR = os.path.dirname(__file__)

QUESTIONS_FILE = os.path.join(THIS_DIR, "testset1_10_questions.jsonl")
ANSWERS_FILE   = os.path.join(THIS_DIR, "testset1_10_answers.jsonl")

MODEL_NAME = "gpt-4o"     # ← BACK TO WORKING MODEL
ANSWER_VOTES = 1


# -------------------------
# JSONL helpers
# -------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_answer_key(path: str) -> Dict[int, str]:
    key = {}
    for ex in read_jsonl(path):
        qid = ex.get("id", ex.get("line"))
        ans = (ex.get("answer_idx") or "").strip().upper()
        if isinstance(qid, int) and re.fullmatch(r"[A-E]", ans):
            key[qid] = ans
    return key


def format_question_with_options(ex: Dict[str, Any]) -> str:
    q = ex.get("question", "").strip()
    opts = ex.get("options", {})
    lines = [q, "Options:"]
    for k in ["A", "B", "C", "D", "E"]:
        if k in opts:
            lines.append(f"{k}. {opts[k]}")
    return "\n".join(lines)


# -------------------------
# OpenAI call (CHAT COMPLETIONS)
# -------------------------
def llm_call(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# -------------------------
# Baseline answering
# -------------------------
def answer_mcq(question_with_options: str) -> str:
    system_prompt = (
        "You are answering a medical multiple-choice question.\n"
        "Return ONLY the option letter (A/B/C/D/E)."
    )
    out = llm_call(system_prompt, question_with_options).upper()
    m = re.search(r"\b([A-E])\b", out)
    return m.group(1) if m else "A"


def answer_consensus(question_with_options: str, n: int) -> str:
    votes = [answer_mcq(question_with_options) for _ in range(n)]
    return Counter(votes).most_common(1)[0][0]


# -------------------------
# Metrics
# -------------------------
def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true + y_pred))
    f1s = []
    for l in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == l)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yp == l and yt != l)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == l and yp != l)
        if tp == 0:
            f1s.append(0.0)
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1s.append(2 * p * r / (p + r))
    return sum(f1s) / len(f1s) if f1s else 0.0


# -------------------------
# Main
# -------------------------
def main():
    answers = load_answer_key(ANSWERS_FILE)

    y_true, y_pred = [], []

    for ex in read_jsonl(QUESTIONS_FILE):
        qid = ex.get("id", ex.get("line"))
        gold = answers.get(qid)
        q_text = format_question_with_options(ex)
        pred = answer_consensus(q_text, ANSWER_VOTES)

        if gold:
            y_true.append(gold)
            y_pred.append(pred)

        print(f"id={qid} | gold={gold} | pred={pred}")

    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
    f1 = macro_f1(y_true, y_pred)

    print("\nAccuracy:", round(acc, 3))
    print("Macro-F1:", round(f1, 3))


if __name__ == "__main__":
    main()
