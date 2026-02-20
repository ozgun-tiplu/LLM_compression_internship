# llm_compression/step1_2_build_token_sheet.py
#
# STEP 1.2 (expensive, incremental, safer):
# - Read a question set
# - Run R1 -> R2 -> R4 (no back-translation)
# - Strong dedupe-by-meaning + DSL rewrite to canonical token
# - Incrementally UPDATE (not overwrite) token_sheet_v6.json
# - ALSO write a compressed dataset (id, compressed, options) like build_dsl_vocab.py
# - Adds local token counts (tb/ta) + periodic progress reporting
# - Adds robust try/except around JSON extraction (prevents early crash -> "short sheet")
# - Atomic writes for token sheet (prevents corruption)
#
# Outputs:
# - token_sheet_v6.json
# - compressed_train_v6.jsonl

import os
import json
import re
from typing import Dict, Any, Iterable, Tuple, List, Set, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from parser.parser import is_valid_dsl


# -------------------------
# ENV + OpenAI client
# -------------------------
ROOT_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=ROOT_ENV, override=True)

client = OpenAI()

# Match build_dsl_vocab's default for richer vocab induction.
# You can change to "gpt-4o-mini" if you explicitly want cheaper step1.
MODEL_NAME = "gpt-4o"


# -------------------------
# Paths
# -------------------------
THIS_DIR = os.path.dirname(__file__)
TRAIN_QUESTIONS_FILE = os.path.join(THIS_DIR, "trainset_sample_20_questions.jsonl.jsonl")

TOKEN_SHEET_FILE = os.path.join(THIS_DIR, "token_sheet_trainset_sample_20_questions.json")
COMPRESSED_TRAIN_FILE = os.path.join(THIS_DIR, "compressed_trainset_sample_20_questions.jsonl")


# -------------------------
# Strict wrapper
# -------------------------
BANNED_TOKENS = {
    "DX", "NEXT", "BUG", "TRAVEL", "COURT", "CONSENT", "DELAY", "TEST", "MOA",
    "ORAL", "WIPED", "LOOPS", "NIGHT",
    "LABVAL", "MANAGE", "NEXTDIAG",
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
# Local token counter (build_dsl_vocab-style)
# -------------------------
def local_token_count(text: str) -> int:
    return len(re.findall(r"\w+|->|\+|\?", text or ""))


# -------------------------
# IO helpers
# -------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSONL not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {ln} in {path}: {e}") from e
    return rows

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

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

def ensure_unk_token(definition: Dict[str, Any]) -> Dict[str, Any]:
    definition = dict(definition)
    definition.setdefault("tokens", {})
    if "UNK" not in definition["tokens"]:
        definition["tokens"]["UNK"] = "unknown/unspecified concept"
    definition.setdefault("operators", {"+": "combines tokens", "->": "relation", "?": "question marker at end"})
    definition.setdefault("examples", {"Example": "A + B ?"})
    return definition

def save_token_sheet_atomic(path: str, definition: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(definition, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_token_sheet(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    data.setdefault("tokens", {})
    return data


# -------------------------
# LLM call
# -------------------------
def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()

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
# Dedupe-by-meaning + DSL rewrite to canonical
# -------------------------
def _norm_meaning(m: str) -> str:
    m = (m or "").strip().lower()
    m = re.sub(r"[\u2019']", "'", m)
    m = re.sub(r"[^a-z0-9%+\- ]+", " ", m)
    m = re.sub(r"\s+", " ", m).strip()
    return m

def _dsl_tokens_and_ops(expr: str) -> List[str]:
    return re.findall(r"[A-Z0-9]{2,8}|->|\+|\?", (expr or "").strip())

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


# -------------------------
# R1 / R2 / R4
# -------------------------
def runde1_medqa(question_text: str, definition: dict, max_retries: int = 3) -> Tuple[str, int, int]:
    system_prompt = (
        "You compress medical multiple-choice questions into a symbolic DSL.\n"
        "GOAL: keep ONLY discriminative clinical clues needed to pick the correct option.\n"
        "\n"
        "RULES:\n"
        "- Tokens MUST be UPPERCASE alphanumeric ONLY.\n"
        "- Token length MUST be 2–8 characters.\n"
        "- Combine tokens using '+'. Optional '->'. Optional '?' only at end.\n"
        "- NO natural language words.\n"
        "\n"
        "AGE RULE:\n"
        "- If the question includes 'X-year-old', you MUST include token AGE<X>.\n"
        "\n"
        "DISCRIMINATIVE SIGNAL:\n"
        "- Prefer specific diagnoses/pathogens/drugs/imaging findings/key labs/vitals.\n"
        "- Avoid generic/meta tokens (LABVAL, MANAGE, NEXTDIAG).\n"
        "\n"
        "REUSE:\n"
        "- If a concept already exists in the sheet, reuse its token; do NOT create synonym tokens.\n"
        "\n"
        "Output ONLY the DSL string."
    )

    user_prompt = f"Compress this into DSL:\n{question_text}"
    tb = local_token_count(question_text)
    last = "UNK?"

    for _ in range(max_retries):
        out = llm_call(system_prompt, user_prompt, temperature=0.2).strip()
        last = out
        ok, _ = strict_dsl_ok(out, definition, question_text)
        if ok:
            return out, tb, local_token_count(out)
        user_prompt = f"INVALID DSL: '{out}'\nFix it. Output ONLY a VALID DSL string."
    return last, tb, local_token_count(last)

def runde2_opt(question_text: str, qc1: str, definition: dict) -> Tuple[str, dict, int]:
    system_prompt = (
        "You refine a compressed DSL string and MUST maintain a token definition sheet.\n"
        "\n"
        "Hard requirements:\n"
        "- Translation must be valid DSL.\n"
        "- Tokens UPPERCASE alphanumeric length 2–8.\n"
        "- If age exists, include AGE##.\n"
        "- NO natural language words in DSL.\n"
        "\n"
        "CRITICAL SHEET RULE:\n"
        "- For EVERY token in Translation, Definition['tokens'] MUST contain an entry.\n"
        "- Keep and extend existing tokens; do not delete prior tokens.\n"
        "\n"
        "REUSE / NO SYNONYMS:\n"
        "- Before creating a new token, check the sheet for an equivalent meaning and reuse.\n"
        "\n"
        "Return STRICT JSON ONLY:\n"
        "{\n"
        '  "Translation": "<dsl>",\n'
        '  "Definition": { "tokens": {...}, "operators": {...}, "examples": {...} }\n'
        "}"
    )

    user_prompt = (
        f"ORIGINAL:\n{question_text}\n\n"
        f"CURRENT DSL:\n{qc1}\n\n"
        f"CURRENT SHEET:\n{json.dumps(definition, ensure_ascii=False)}\n\n"
        "Improve the DSL and update the sheet."
    )

    # Robust fallback like build_dsl_vocab.py (prevents crash -> short sheet)
    try:
        ans = llm_call(system_prompt, user_prompt, temperature=0.1)
        parsed = _extract_json_object(ans)
        qc2 = parsed.get("Translation", qc1)
        new_def = parsed.get("Definition", definition)

        qc2, new_def = dedupe_definition_and_rewrite(qc2, new_def)

        ok, _ = strict_dsl_ok(qc2, new_def, question_text)
        if ok:
            return qc2, new_def, local_token_count(qc2)

        return qc1, definition, local_token_count(qc1)
    except Exception:
        return qc1, definition, local_token_count(qc1)

def runde4_consistency_no_back(original_text: str, qc2: str, definition: dict) -> Tuple[str, dict]:
    system_prompt = (
        "You check whether the DSL preserves the original medical question.\n"
        "If meaning drift happened, adjust the DSL minimally.\n"
        "Ensure Definition includes meanings for ALL tokens used.\n"
        "\n"
        "REUSE / NO SYNONYMS:\n"
        "- Reuse existing tokens for equivalent meanings; do NOT introduce synonym tokens.\n"
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
        "If DSL is wrong, fix it."
    )

    # Robust fallback like build_dsl_vocab.py (prevents crash -> short sheet)
    try:
        ans = llm_call(system_prompt, user_prompt, temperature=0.0)
        parsed = _extract_json_object(ans)
        qc3 = parsed.get("FinalCompressed", qc2)
        new_def = parsed.get("Definition", definition)

        qc3, new_def = dedupe_definition_and_rewrite(qc3, new_def)

        ok, _ = strict_dsl_ok(qc3, new_def, original_text)
        if ok:
            return qc3, new_def

        return qc2, definition
    except Exception:
        return qc2, definition


def main():
    print("TRAIN_QUESTIONS_FILE   =", TRAIN_QUESTIONS_FILE)
    print("TOKEN_SHEET_FILE       =", TOKEN_SHEET_FILE)
    print("COMPRESSED_TRAIN_FILE  =", COMPRESSED_TRAIN_FILE)
    print("MODEL                  =", MODEL_NAME)

    existing = load_token_sheet(TOKEN_SHEET_FILE)
    if existing is None:
        definition = ensure_unk_token({
            "tokens": {},
            "operators": {"+": "combines tokens", "->": "relation", "?": "question marker at end"},
            "examples": {"Example": "A + B ?"},
        })
    else:
        definition = ensure_unk_token(existing)

    print(f"Loaded tokens: {len(definition.get('tokens', {}))}")

    examples = read_jsonl(TRAIN_QUESTIONS_FILE)
    compressed_rows: List[Dict[str, Any]] = []

    for i, ex in enumerate(tqdm(examples, desc="Building token sheet (v6)", unit="q"), 1):
        qid = ex.get("id")
        options = ex.get("options") or {}

        printable_original = format_question_with_options(ex)

        qc1, tb, ta1 = runde1_medqa(printable_original, definition)
        qc2, definition, ta2 = runde2_opt(printable_original, qc1, definition)
        qc3, definition = runde4_consistency_no_back(printable_original, qc2, definition)

        definition = ensure_unk_token(definition)

        compressed_rows.append({
            "id": qid,
            "compressed": qc3,
            "options": options,
        })

        # build_dsl_vocab-style progress + local token counts
        if i % 25 == 0:
            tqdm.write(
                f"Processed {i} questions... tokens so far: {len(definition.get('tokens', {}))} | "
                f"local tb={tb}, ta1={ta1}, ta2={ta2}"
            )

    # Save outputs
    save_token_sheet_atomic(TOKEN_SHEET_FILE, definition)
    write_jsonl(COMPRESSED_TRAIN_FILE, compressed_rows)

    print("Saved token sheet to:", TOKEN_SHEET_FILE)
    print("Saved compressed train set to:", COMPRESSED_TRAIN_FILE)
    print("Final token count:", len(definition.get("tokens", {})))


if __name__ == "__main__":
    main()
