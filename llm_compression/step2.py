# llm_compression/step2_map_and_answer_cached_update.py
#
# STEP 2 (cheaper, cached; updates token sheet incrementally):
#
# Goal:
# - Load token_sheet_v6.json once
# - For each new question:
#     A) Map question -> DSL using ONLY existing tokens (cheap-ish)
#     B) Answer using DSL + options (cheap-ish)
#     C) If mapping is too lossy (UNK-heavy), optionally extend the sheet:
#          - Run a constrained "add tokens" update (R2-style) on that ONE question
#          - Merge + dedupe + rewrite to canonical
#          - Persist token_sheet_v6.json
#       Then remap + answer using the updated sheet.
#
# Key design: "cached version"
# - The sheet is loaded once and held in memory.
# - We DO NOT re-send the full sheet as JSON for every question.
# - Instead we send only:
#     - AllowedTokens list (always)
#     - A small "token glossary" subset needed for the produced DSL (answering)
# - When sheet grows, it is saved back to token_sheet_v6.json.
#
# Inputs:
# - NEW_QUESTIONS_FILE: a new set of questions JSONL (must include id, question, options)
# - NEW_ANSWERS_FILE  : optional, for evaluation (id, answer_idx)
#
# Outputs:
# - mapped_and_answered.jsonl with id, dsl, pred, gold (if provided), and whether sheet was updated.

import os
import json
import re
import time
from typing import Dict, Any, Iterable, List, Set, Optional, Tuple
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from parser.parser import is_valid_dsl


# -------------------------
# PATHS (change only these)
# -------------------------
THIS_DIR = os.path.dirname(__file__)

NEW_QUESTIONS_FILE = os.path.join(THIS_DIR, "test_10_questions.jsonl")
NEW_ANSWERS_FILE   = os.path.join(THIS_DIR, "test_10_answers.jsonl")  # optional; can be missing

TOKEN_SHEET_FILE   = os.path.join(THIS_DIR, "token_sheet_trainset_sample_20_questions.json")

OUT_FILE           = os.path.join(THIS_DIR, "mapped_and_answered_cached_update.jsonl")


# -------------------------
# ENV + OpenAI client
# -------------------------
ROOT_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=ROOT_ENV, override=True)

client = OpenAI(timeout=90.0)

MODEL_NAME = "gpt-4o-mini"
ANSWER_VOTES = 1  # 1 cheapest, 3–5 more stable

MAX_RETRIES = 5
BACKOFF_BASE = 1.6


# -------------------------
# Strict wrapper + token rules (same family as step 1)
# -------------------------
BANNED_TOKENS = {
    "DX", "NEXT", "BUG", "TRAVEL", "COURT", "CONSENT", "DELAY", "TEST", "MOA",
    "ORAL", "WIPED", "LOOPS", "NIGHT",
    "LABVAL", "MANAGE", "NEXTDIAG",
}
AGE_RE = re.compile(r"\b(\d{1,3})-year-old\b", re.IGNORECASE)


# -------------------------
# IO helpers
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

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def try_load_answers(path: str) -> Dict[int, str]:
    if not os.path.isfile(path):
        return {}
    key: Dict[int, str] = {}
    for ex in read_jsonl(path):
        qid = get_qid(ex)
        if qid is None:
            continue
        letter = (ex.get("answer_idx") or "").strip().upper()
        if re.fullmatch(r"[A-E]", letter or ""):
            key[qid] = letter
    return key

def get_qid(ex: Dict[str, Any]) -> Optional[int]:
    v = ex.get("id")
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None

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
# DSL helpers
# -------------------------
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
# Dedupe-by-meaning + DSL rewrite (same as step 1)
# -------------------------
def _norm_meaning(m: str) -> str:
    m = (m or "").strip().lower()
    m = re.sub(r"[\u2019']", "'", m)
    m = re.sub(r"[^a-z0-9%+\- ]+", " ", m)
    m = re.sub(r"\s+", " ", m).strip()
    return m

def _dsl_tokens_and_ops(expr: str) -> List[str]:
    return re.findall(r"[A-Z0-9]{2,8}|->|\+|\?", expr.strip())

def _rewrite_dsl_tokens(expr: str, mapping: Dict[str, str]) -> str:
    parts = _dsl_tokens_and_ops(expr)
    out_parts: List[str] = []
    for p in parts:
        if re.fullmatch(r"[A-Z0-9]{2,8}", p):
            out_parts.append(mapping.get(p, p))
        else:
            out_parts.append(p)
    return "".join(out_parts)

def dedupe_definition_and_rewrite(expr: str, definition: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    definition = dict(definition)
    toks: Dict[str, str] = dict(definition.get("tokens", {}) or {})

    canonical_by_meaning: Dict[str, str] = {}
    token_to_canonical: Dict[str, str] = {}

    for tok, meaning in toks.items():
        nm = _norm_meaning(str(meaning))
        if not nm:
            token_to_canonical[tok] = tok
            continue
        if nm not in canonical_by_meaning:
            canonical_by_meaning[nm] = tok
            token_to_canonical[tok] = tok
        else:
            token_to_canonical[tok] = canonical_by_meaning[nm]

    rewritten = _rewrite_dsl_tokens(expr, token_to_canonical)

    new_tokens: Dict[str, str] = {}
    seen_meanings: Set[str] = set()
    for tok, meaning in toks.items():
        nm = _norm_meaning(str(meaning))
        if not nm:
            if tok not in new_tokens:
                new_tokens[tok] = meaning
            continue
        if nm in seen_meanings:
            continue
        seen_meanings.add(nm)
        canon = canonical_by_meaning.get(nm, tok)
        if canon == tok:
            new_tokens[tok] = meaning
        else:
            if canon not in new_tokens:
                new_tokens[canon] = toks.get(canon, meaning)

    definition["tokens"] = new_tokens
    return rewritten, definition


def save_token_sheet_atomic(path: str, definition: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(definition, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -------------------------
# LLM call with retries
# -------------------------
def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF_BASE ** attempt)
    raise RuntimeError(f"llm_chat failed after {MAX_RETRIES} retries: {last_err}")


# -------------------------
# Cache strategy:
# - Do NOT send full token sheet every time.
# - Send AllowedTokens list for mapping.
# - For answering, send only definitions for tokens used in DSL.
# -------------------------
def build_glossary_for_dsl(definition: Dict[str, Any], dsl: str) -> Dict[str, str]:
    toks = tokens_in_dsl(dsl)
    sheet_tokens = definition.get("tokens", {}) or {}
    gloss: Dict[str, str] = {}
    for t in sorted(toks):
        if t in sheet_tokens:
            gloss[t] = sheet_tokens[t]
    # Always include UNK if used
    if "UNK" in toks and "UNK" not in gloss and "UNK" in sheet_tokens:
        gloss["UNK"] = sheet_tokens["UNK"]
    return gloss


# -------------------------
# (A) Mapping using ONLY existing tokens - send only AllowedTokens (token names), not the full sheet.
# -------------------------
def map_to_existing_tokens(
    allowed_tokens: Set[str],
    frozen_definition: Dict[str, Any],
    question_text: str,
    max_retries: int = 2,
) -> str:
    allowed_list = ", ".join(sorted(allowed_tokens))

    system_prompt = (
        "Map the medical question into the existing DSL.\n"
        "CRITICAL: You may ONLY use tokens from AllowedTokens. Do NOT invent new tokens.\n"
        "Tokens must be UPPERCASE alphanumeric length 2–8.\n"
        "Combine tokens with '+'. Optional '->'. Optional '?' only at end.\n"
        "If you cannot express a concept with AllowedTokens, use UNK.\n"
        "Output ONLY the DSL string."
    )

    last = "UNK?"
    for _ in range(max_retries):
        out = llm_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"AllowedTokens:\n{allowed_list}"},
                {"role": "user", "content": f"Question:\n{question_text}"},
            ],
            temperature=0.0,
        ).strip()
        last = out

        if not re.fullmatch(r"[A-Z0-9+\-\>\? ]+", out or ""):
            continue

        toks = tokens_in_dsl(out)
        if not toks or any(t not in allowed_tokens for t in toks):
            continue

        if not is_valid_dsl(out, frozen_definition):
            continue

        return out

    return last


# -------------------------
# (B) Answer using DSL + SMALL glossary, send only a small glossary for tokens used in the DSL.
# -------------------------
def _parse_letter_strict(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip().upper()
    if re.fullmatch(r"[A-E]", s):
        return s
    m = re.search(r"\b([A-E])\b", s)
    if m:
        return m.group(1)
    if s and s[0] in "ABCDE":
        return s[0]
    return None

def answer_mcq_with_glossary(glossary: Dict[str, str], dsl: str, options: Dict[str, str]) -> str:
    system_prompt = (
        "You answer a medical multiple-choice question.\n"
        "You are given a DSL representation + a TOKEN GLOSSARY + options.\n"
        "CRITICAL OUTPUT RULE:\n"
        "- Output EXACTLY ONE character: A, B, C, D, or E.\n"
        "- No punctuation, no spaces, no newline, no explanation.\n"
    )

    gloss_text = json.dumps(glossary, ensure_ascii=False, sort_keys=True)

    opts_lines = [f"{k}. {options[k]}" for k in ["A", "B", "C", "D", "E"] if k in options]
    user_q = (
        f"TOKEN_GLOSSARY_JSON:\n{gloss_text}\n\n"
        f"DSL:\n{dsl}\n\n"
        f"Options:\n{chr(10).join(opts_lines)}\n\n"
        "Answer with ONE letter only."
    )

    out = llm_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_q},
        ],
        temperature=0.0,
    )
    letter = _parse_letter_strict(out)
    if letter:
        return letter

    # retry once
    out2 = llm_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_q},
            {"role": "user", "content": f"INVALID OUTPUT: {out!r}\nReturn EXACTLY ONE LETTER A B C D E."},
        ],
        temperature=0.0,
    )
    letter2 = _parse_letter_strict(out2)
    return letter2 if letter2 else "NA"

def consensus_answer(glossary: Dict[str, str], dsl: str, options: Dict[str, str], n: int) -> str:
    votes = [answer_mcq_with_glossary(glossary, dsl, options) for _ in range(n)]
    return Counter(votes).most_common(1)[0][0]


# -------------------------
# (C) Optional on-the-fly token sheet expansion for UNK-heavy mapping
#     This is the only time we send the (full) sheet, and only on problematic questions.
# -------------------------
def should_expand_sheet(dsl: str, unk_threshold: float = 0.34) -> bool:
    toks = list(tokens_in_dsl(dsl))
    if not toks:
        return True
    unk_count = sum(1 for t in toks if t == "UNK")
    return (unk_count / max(1, len(toks))) >= unk_threshold

def expand_sheet_for_question(question_text: str, current_dsl: str, definition: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Similar to Step 1 R2: allow new tokens + definitions, with reuse/no-synonyms instruction,
    then dedupe+rewrite. Used sparingly for cost control.
    """
    system_prompt = (
        "You improve a DSL translation and MAY introduce NEW tokens if truly necessary.\n"
        "You MUST maintain the token definition sheet.\n"
        "\n"
        "Hard requirements:\n"
        "- Output DSL using only UPPERCASE alphanumeric tokens length 2–8 and operators +, ->, ?.\n"
        "- If age exists in original, include AGE##.\n"
        "- NO natural language words inside DSL.\n"
        "\n"
        "REUSE / NO SYNONYMS:\n"
        "- Before creating any new token, search the existing sheet for equivalent meaning and reuse.\n"
        "\n"
        "SHEET RULE:\n"
        "- For every token used in Translation, Definition['tokens'] must include an entry.\n"
        "\n"
        "Return STRICT JSON ONLY:\n"
        "{\n"
        '  "Translation": "<dsl>",\n'
        '  "Definition": { "tokens": {...}, "operators": {...}, "examples": {...} }\n'
        "}"
    )

    user_prompt = (
        f"ORIGINAL:\n{question_text}\n\n"
        f"CURRENT DSL:\n{current_dsl}\n\n"
        f"CURRENT SHEET:\n{json.dumps(definition, ensure_ascii=False)}\n\n"
        "Improve the DSL for answerability and update the sheet. Add new tokens only if unavoidable."
    )

    ans = llm_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    parsed = _extract_json_object(ans)
    qc2 = parsed.get("Translation", current_dsl)
    new_def = parsed.get("Definition", definition)

    qc2, new_def = dedupe_definition_and_rewrite(qc2, new_def)

    ok, _ = strict_dsl_ok(qc2, new_def, question_text)
    if ok:
        return qc2, new_def
    return current_dsl, definition

def _extract_json_object(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
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
# Metrics (optional)
# -------------------------
def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set([x for x in y_true if x] + [x for x in y_pred if x]))
    if not labels:
        return 0.0

    def f1_for_label(lbl: str) -> float:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lbl and yp == lbl)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp != lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return sum(f1_for_label(l) for l in labels) / len(labels)


def main():
    print("NEW_QUESTIONS_FILE =", NEW_QUESTIONS_FILE)
    print("NEW_ANSWERS_FILE   =", NEW_ANSWERS_FILE if os.path.isfile(NEW_ANSWERS_FILE) else "(missing/skip scoring)")
    print("TOKEN_SHEET_FILE   =", TOKEN_SHEET_FILE)
    print("MODEL              =", MODEL_NAME)
    print("ANSWER_VOTES       =", ANSWER_VOTES)

    if not os.path.isfile(TOKEN_SHEET_FILE):
        raise FileNotFoundError(f"Token sheet not found: {TOKEN_SHEET_FILE}. Run step1_build_token_sheet.py first.")

    with open(TOKEN_SHEET_FILE, "r", encoding="utf-8") as f:
        definition = json.load(f)

    definition.setdefault("tokens", {})
    if "UNK" not in definition["tokens"]:
        definition["tokens"]["UNK"] = "unknown/unspecified concept"
    definition.setdefault("operators", {"+": "combines tokens", "->": "relation", "?": "question marker at end"})
    definition.setdefault("examples", {"Example": "A + B ?"})

    # In-memory cache: allowed token set
    allowed_tokens: Set[str] = set(definition["tokens"].keys())

    # Optional gold answers
    answer_key = try_load_answers(NEW_ANSWERS_FILE)

    rows = list(read_jsonl(NEW_QUESTIONS_FILE))
    out_rows: List[Dict[str, Any]] = []

    y_true: List[str] = []
    y_pred: List[str] = []

    for ex in tqdm(rows, desc="Map+Answer (cached+update)", unit="q"):
        qid = get_qid(ex)
        if qid is None:
            continue

        options = ex.get("options") or {}
        question_text = format_question_with_options(ex)
        gold = answer_key.get(qid, "")

        # 1) Map with existing tokens only (cheap)
        dsl = map_to_existing_tokens(
            allowed_tokens=allowed_tokens,
            frozen_definition=definition,
            question_text=question_text,
            max_retries=2,
        )

        updated_sheet = False

        # 2) If too many UNKs, expand sheet for this question (expensive, but only sometimes)
        if should_expand_sheet(dsl, unk_threshold=0.34):
            dsl2, definition2 = expand_sheet_for_question(question_text, dsl, definition)
            if definition2 is not definition:
                definition = definition2
            if dsl2 != dsl:
                dsl = dsl2
            # Update caches
            allowed_tokens = set((definition.get("tokens") or {}).keys())
            updated_sheet = True
            save_token_sheet_atomic(TOKEN_SHEET_FILE, definition)

            # Remap with new tokens only if we still have invalid DSL somehow (rare)
            ok, _ = strict_dsl_ok(dsl, definition, question_text)
            if not ok:
                dsl = map_to_existing_tokens(allowed_tokens, definition, question_text, max_retries=2)

        # 3) Answer using ONLY the small glossary for tokens in DSL (cheap)
        glossary = build_glossary_for_dsl(definition, dsl)
        pred = consensus_answer(glossary, dsl, options, n=ANSWER_VOTES)

        if gold:
            y_true.append(gold)
            y_pred.append(pred)

        out_rows.append({
            "id": qid,
            "dsl": dsl,
            "pred": pred,
            "gold": gold,
            "updated_sheet": updated_sheet,
        })

    write_jsonl(OUT_FILE, out_rows)
    print("Saved outputs to:", OUT_FILE)
    print("Final token count:", len((definition.get("tokens") or {})))

    if y_true and y_pred and len(y_true) == len(y_pred):
        acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
        f1 = macro_f1(y_true, y_pred)
        print("\n=== Score Summary ===")
        print(f"Questions scored: {len(y_true)}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Macro F1 (A–E): {f1:.3f}")
    else:
        print("\n=== Score Summary ===")
        print("No gold labels found (or mismatch), so no scoring reported.")


if __name__ == "__main__":
    main()
