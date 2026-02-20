
#  (NO back-translation):
# - Separate questions + answers JSONL (prevents leakage)
# - Rounds 1–2 to compress + maintain token sheet
# - Round 4 consistency check WITHOUT back-translation
# - Round 5 answer from DSL + sheet + options
# - Compute Accuracy + Macro-F1


import os
import json
import re
import time
from typing import Dict, Any, Iterable, Tuple, List, Set
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

from parser.parser import is_valid_dsl


# -------------------------
# ENV + OpenAI client
# -------------------------
ROOT_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=ROOT_ENV, override=True)

# Add timeout so it doesn't hang forever
client = OpenAI(timeout=90.0, max_retries=0)

MODEL_NAME = "gpt-4.1"
ANSWER_VOTES = 1  # 1 cheapest, 3–5 more stable


# -------------------------
# Local input (separate questions + answers to prevent leakage)
# -------------------------
THIS_DIR = os.path.dirname(__file__)
QUESTIONS_FILE = os.path.join(THIS_DIR, "testset1_10_questions.jsonl")
ANSWERS_FILE   = os.path.join(THIS_DIR, "testset1_10_answers.jsonl")

# -------------------------
# Token sheet output (persisted across runs)
# -------------------------
TOKEN_SHEET_OUT = os.path.join(THIS_DIR, "token_sheet_generated.json")


# -------------------------
# Strict wrapper validator (MedQA-only)
# -------------------------
BANNED_TOKENS = {
    "DX", "NEXT", "BUG", "TRAVEL", "COURT", "CONSENT", "DELAY", "TEST", "MOA",
    "ORAL", "WIPED", "LOOPS", "NIGHT"
}
AGE_RE = re.compile(r"\b(\d{1,3})-year-old\b", re.IGNORECASE)

def tokens_in_dsl(expr: str) -> Set[str]:
    expr = expr.replace("->", "+").replace("?", "")
    parts = [p.strip() for p in expr.split("+") if p.strip()]
    return set(parts)

def strict_dsl_ok(expr: str, definition: Dict[str, Any], original_text: str) -> Tuple[bool, str]:
    if not is_valid_dsl(expr, definition):
        return False, "parse_dsl rejected"

    toks = tokens_in_dsl(expr)

    for t in toks:
        if not re.fullmatch(r"[A-Z0-9]{2,8}", t):
            return False, f"bad token format: {t}"
        if t in BANNED_TOKENS:
            return False, f"banned token: {t}"

    m = AGE_RE.search(original_text)
    if m:
        age = m.group(1)
        if f"AGE{age}" not in toks:
            return False, f"missing/incorrect AGE token (need AGE{age})"

    return True, "ok"


# -------------------------
# Tokenizer (simple + stable)
# -------------------------
def tokenize(text: str) -> int:
    return len(re.findall(r"\w+|->|\+|\?", text))


# -------------------------
# Token sheet persistence
# -------------------------
def default_token_sheet() -> Dict[str, Any]:
    return {
        "tokens": {},
        "operators": {"+": "combines tokens", "->": "relation", "?": "question marker at end"},
        "examples": {"Example": "A + B ?"},
    }

def load_token_sheet(path: str) -> Dict[str, Any]:
    """
    Load an existing token sheet so runs can resume and keep extending tokens.
    If missing or invalid, returns a clean default sheet.
    """
    if not os.path.isfile(path):
        return default_token_sheet()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # minimal structural sanity checks
        if not isinstance(data, dict):
            return default_token_sheet()
        if "tokens" not in data or not isinstance(data.get("tokens"), dict):
            data["tokens"] = {}
        if "operators" not in data or not isinstance(data.get("operators"), dict):
            data["operators"] = default_token_sheet()["operators"]
        if "examples" not in data or not isinstance(data.get("examples"), dict):
            data["examples"] = default_token_sheet()["examples"]
        return data
    except Exception:
        return default_token_sheet()

def save_token_sheet(definition: Dict[str, Any], path: str) -> None:
    """
    Atomic-ish write to avoid corrupting the sheet on crash/interruption.
    Writes to <path>.tmp then replaces.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(definition, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


# -------------------------
# OpenAI call (with retry)
# -------------------------
def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except KeyboardInterrupt:
            raise
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return ""


def _extract_json_object(s: str) -> Dict[str, Any]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end + 1])

    raise ValueError("Could not parse JSON from model output.")


# -------------------------
# IO: JSONL helpers
# -------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSONL not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {ln} in {path}: {e}") from e


def load_answer_key(path: str) -> Dict[int, str]:
    """
    Load gold answer letters from answers JSONL.
    Expected per line: {"id": <int>, "answer_idx": "A".."E"} OR {"line": <int>, ...}
    Returns mapping: {id: "A".."E"}
    """
    key: Dict[int, str] = {}
    for ex in read_jsonl(path):
        qid = ex.get("id", ex.get("line"))
        if isinstance(qid, str) and qid.isdigit():
            qid = int(qid)
        letter = (ex.get("answer_idx") or "").strip().upper()
        if isinstance(qid, int) and re.fullmatch(r"[A-E]", letter or ""):
            key[qid] = letter
    return key


def format_question_with_options(ex: Dict[str, Any]) -> str:
    q = (ex.get("question") or "").strip()
    opts = ex.get("options") or {}
    parts = [q]
    if isinstance(opts, dict) and opts:
        parts.append("Options:")
        for k in ["A", "B", "C", "D", "E"]:
            if k in opts:
                parts.append(f"{k}. {str(opts[k]).strip()}")
    return "\n".join(parts).strip()


# -------------------------
# Round 1: invent DSL (MedQA) + strict wrapper
# -------------------------
def runde1_medqa(question_text: str, definition: dict, max_retries: int = 3) -> Tuple[str, int, int]:
    system_prompt = (
        "You compress medical multiple-choice questions into a symbolic DSL.\n"
        "GOAL: keep ONLY discriminative clinical clues needed to pick the correct option.\n"
        "\n"
        "RULES:\n"
        "- Tokens MUST be UPPERCASE alphanumeric ONLY.\n"
        "- Token length MUST be 2–8 characters.\n"
        "- Combine tokens using '+'.\n"
        "- OPTIONAL: use '->' for relation between evidence and candidates.\n"
        "- OPTIONAL: put '?' only at the very end.\n"
        "- NO natural language words.\n"
        "- NO symbols except letters, digits, '+', '->', '?'.\n"
        "\n"
        "AGE RULE:\n"
        "- If the question includes 'X-year-old', you MUST include token AGE<X> (e.g., AGE15, AGE4).\n"
        "\n"
        "PRACTICAL:\n"
        "- Prefer medical abbreviations when appropriate (e.g., TSS, HUS, TTP, KOH, NSCLC, TYPH).\n"
        "- Avoid vague tokens (NO: DX, NEXT, BUG, TRAVEL, TEST, MOA).\n"
        "\n"
        "Output ONLY the DSL string."
    )

    user_prompt = f"Compress this into DSL:\n{question_text}"
    tb = tokenize(question_text)
    last = ""

    for _ in range(max_retries):
        out = llm_call(system_prompt, user_prompt, temperature=0.2).strip()
        last = out

        ok, _why = strict_dsl_ok(out, definition, question_text)
        if ok:
            return out, tb, tokenize(out)

        user_prompt = (
            f"INVALID DSL: '{out}'\n"
            "Fix it. Output ONLY a VALID DSL string following the rules."
        )

    return last, tb, tokenize(last)


# -------------------------
# Round 2: optimize + update sheet
# -------------------------
def runde2_opt(question_text: str, qc1: str, definition: dict) -> Tuple[str, dict, int]:
    system_prompt = (
        "You refine a compressed DSL string and MUST maintain a token definition sheet.\n"
        "\n"
        "Hard requirements:\n"
        "- Your Translation must be valid DSL (same token rules as before).\n"
        "- Tokens must be UPPERCASE alphanumeric, length 2–8.\n"
        "- If age exists in the original, include AGE##.\n"
        "- NO natural language words in the DSL.\n"
        "\n"
        "CRITICAL SHEET RULE:\n"
        "- For EVERY token that appears in Translation, Definition['tokens'] MUST contain an entry.\n"
        "- Meanings should be short medical phrases.\n"
        "- Keep and extend existing tokens; do not delete prior tokens.\n"
        "\n"
        "Return STRICT JSON ONLY:\n"
        "{\n"
        '  "Translation": "<dsl>",\n'
        '  "Definition": {\n'
        '     "tokens": { "TOK":"meaning", ... },\n'
        '     "operators": { "+":"...", "->":"...", "?":"..." },\n'
        '     "examples": { ... }\n'
        "  }\n"
        "}"
    )

    user_prompt = (
        f"ORIGINAL:\n{question_text}\n\n"
        f"CURRENT DSL:\n{qc1}\n\n"
        f"CURRENT SHEET:\n{json.dumps(definition, ensure_ascii=False)}\n\n"
        "Improve the DSL for answerability and update the sheet accordingly."
    )

    ans = llm_call(system_prompt, user_prompt, temperature=0.1)

    try:
        parsed = _extract_json_object(ans)
        qc2 = parsed.get("Translation", qc1)
        new_def = parsed.get("Definition", definition)

        ok, _why = strict_dsl_ok(qc2, new_def, question_text)
        if ok:
            return qc2, new_def, tokenize(qc2)
        return qc1, definition, tokenize(qc1)
    except Exception:
        return qc1, definition, tokenize(qc1)


# -------------------------
# Round 4: consistency check WITHOUT back-translation
# -------------------------
def runde4_consistency_no_back(original_text: str, qc2: str, definition: dict) -> Tuple[str, dict]:
    system_prompt = (
        "You check whether the DSL preserves the original medical question.\n"
        "If meaning drift happened, adjust the DSL minimally.\n"
        "Do NOT invent natural language words inside the DSL.\n"
        "Ensure Definition includes meanings for ALL tokens used.\n"
        "\n"
        "Return STRICT JSON ONLY:\n"
        "{\n"
        '  "FinalCompressed": "<dsl>",\n'
        '  "Definition": { ... updated sheet ... }\n'
        "}"
    )
    user_prompt = (
        f"ORIGINAL:\n{original_text}\n\n"
        f"CURRENT DSL:\n{qc2}\n\n"
        f"SHEET:\n{json.dumps(definition, ensure_ascii=False)}\n\n"
        "If DSL is wrong, fix it. Also ensure Definition includes meanings for ALL tokens used."
    )

    ans = llm_call(system_prompt, user_prompt, temperature=0.0)

    try:
        parsed = _extract_json_object(ans)
        qc3 = parsed.get("FinalCompressed", qc2)
        new_def = parsed.get("Definition", definition)

        ok, _why = strict_dsl_ok(qc3, new_def, original_text)
        if ok:
            return qc3, new_def
        return qc2, definition
    except Exception:
        return qc2, definition


# -------------------------
# Round 5: answer MCQ (A–E) + consensus
# -------------------------
def runde5_answer_mcq(qc3: str, definition: dict, options: Dict[str, str]) -> str:
    system_prompt = (
        "You answer a medical multiple-choice question.\n"
        "You are given a DSL representation + a token sheet + options.\n"
        "Return ONLY the option letter (A/B/C/D/E)."
    )

    opts_lines = []
    for k in ["A", "B", "C", "D", "E"]:
        if k in options:
            opts_lines.append(f"{k}. {options[k]}")
    opts_text = "\n".join(opts_lines)

    user_prompt = (
        f"DSL:\n{qc3}\n\n"
        f"Sheet:\n{json.dumps(definition, ensure_ascii=False)}\n\n"
        f"Options:\n{opts_text}\n\n"
        "Answer with ONE letter only."
    )

    out = llm_call(system_prompt, user_prompt, temperature=0.0).strip().upper()
    if re.fullmatch(r"[A-E]", out):
        return out
    m = re.search(r"\b([A-E])\b", out)
    return m.group(1) if m else "A"


def runde5_consensus_mcq(qc3: str, definition: dict, options: Dict[str, str], n: int) -> str:
    answers = [runde5_answer_mcq(qc3, definition, options) for _ in range(n)]
    return Counter(answers).most_common(1)[0][0]


# -------------------------
# Metrics: macro-F1 for letters (A–E)
# -------------------------
def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set([x for x in y_true if x] + [x for x in y_pred if x]))
    if not labels:
        return 0.0

    def f1_for_label(lbl: str) -> float:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lbl and yp == lbl)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp != lbl)
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return sum(f1_for_label(l) for l in labels) / len(labels)


# -------------------------
# Main
# -------------------------
def main():
    print("QUESTIONS_FILE =", QUESTIONS_FILE)
    print("ANSWERS_FILE   =", ANSWERS_FILE)
    print("MODEL          =", MODEL_NAME)
    print("ANSWER_VOTES   =", ANSWER_VOTES)
    print("TOKEN_SHEET_OUT =", TOKEN_SHEET_OUT)

    answer_key = load_answer_key(ANSWERS_FILE)

    # Load prior sheet if it exists; otherwise start a clean default one.
    definition = load_token_sheet(TOKEN_SHEET_OUT)

    y_true: List[str] = []
    y_pred: List[str] = []

    for i, ex in enumerate(read_jsonl(QUESTIONS_FILE), 1):
        qid = ex.get("id", ex.get("line"))
        if isinstance(qid, str) and qid.isdigit():
            qid = int(qid)

        options = ex.get("options") or {}
        gold_letter = answer_key.get(qid, "")

        if gold_letter:
            y_true.append(gold_letter)

        printable_original = format_question_with_options(ex)

        print(f"\n--- Question {i} (id={qid}) ---")

        qc1, tb, ta1 = runde1_medqa(printable_original, definition)
        print(f"Tokens before: {tb}, after Round1: {ta1}")
        print("Round1 DSL:", qc1)

        qc2, definition, ta2 = runde2_opt(printable_original, qc1, definition)
        print(f"Tokens after Round2: {ta2}")
        print("Round2 DSL:", qc2)

        qc3, definition = runde4_consistency_no_back(printable_original, qc2, definition)
        print("Round4 Final DSL:", qc3)

        pred = runde5_consensus_mcq(qc3, definition, options, n=ANSWER_VOTES)
        y_pred.append(pred)

        print(f"Gold: {gold_letter or 'NA'} | Pred: {pred}")

        # Persist after each question so you never lose progress.
        save_token_sheet(definition, TOKEN_SHEET_OUT)
        print("Saved token sheet to:", TOKEN_SHEET_OUT)

    # Final save (explicit)
    save_token_sheet(definition, TOKEN_SHEET_OUT)
    print("\nFinal token sheet saved to:", TOKEN_SHEET_OUT)

    if y_true and y_pred and len(y_true) == len(y_pred):
        acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
        f1 = macro_f1(y_true, y_pred)
        print("\n=== Summary ===")
        print(f"Questions scored: {len(y_true)}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Macro F1 (A–E): {f1:.3f}")
    else:
        print("\n=== Summary ===")
        print("No gold labels found (or score length mismatch), so no F1/accuracy computed.")
        print("Check that answers JSONL uses matching id/line values and includes answer_idx.")


if __name__ == "__main__":
    main()
