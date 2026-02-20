# parser/parser.py

from __future__ import annotations

from typing import List, Dict, Literal, Optional
from pydantic import BaseModel


class DslParseError(ValueError):
    """Raised when a DSL string does not match the expected grammar."""


class DslExpression(BaseModel):
    """
    Internal representation of a DSL expression.

    kind:
        - "set"      : a simple concept set, optionally a question
        - "relation" : LEFT -> RIGHT, optionally a question
    """
    kind: Literal["set", "relation"]
    left: List[str]              # always present
    right: Optional[List[str]]   # only for kind == "relation"
    is_question: bool
    raw: str


def _strip_question_mark(expr: str) -> tuple[str, bool]:
    """
    Ensure that '?' appears at most once and only at the very end.
    Return (expr_without_q, is_question).
    """
    expr = expr.strip()
    if not expr:
        raise DslParseError("Empty DSL expression.")

    if "?" not in expr:
        return expr, False

    # Only allow '?' at the end, possibly with one space before it.
    if not expr.endswith("?"):
        raise DslParseError("Question mark '?' is only allowed at the end.")

    # Ensure no other '?' inside
    if "?" in expr[:-1]:
        raise DslParseError("Only one '?' is allowed in the expression.")

    expr_wo_q = expr[:-1].rstrip()  # remove trailing '?' and optional space
    if not expr_wo_q:
        raise DslParseError("No content before '?'.")

    return expr_wo_q, True


def _parse_concept_set(s: str, tokens_def: Dict[str, str]) -> List[str]:
    """
    Parse a 'concept set':
        TOKEN
        TOKEN + TOKEN
        TOKEN + TOKEN + TOKEN ...

    - tokens are split by '+'
    - tokens must be [A-Za-z0-9]+
    - optionally: must exist in tokens_def (if tokens_def is non-empty)
    """
    if not s:
        raise DslParseError("Empty concept set.")

    parts = [p.strip() for p in s.split("+")]
    if any(not p for p in parts):
        raise DslParseError("Invalid '+' usage (empty token between '+').")

    result: List[str] = []
    for p in parts:
        # very simple token pattern: letters and digits only
        if not p.isalnum():
            raise DslParseError(f"Invalid token '{p}'. Tokens must be alphanumeric.")

        if tokens_def and p not in tokens_def:
            # you can relax this if you want to allow new tokens
            pass
            # raise DslParseError(f"Unknown token '{p}' (not in definition).")

        result.append(p)

    return result


def parse_dsl(expr: str, definition: Dict) -> DslExpression:
    """
    Parse a DSL expression according to v1 grammar:

    Allowed operators:
        +   : combine tokens
        ->  : relation between two concept sets (LEFT -> RIGHT)
        ?   : question marker, only once at the very end

    Valid shapes:
        SET
        SET ?
        SET -> SET
        SET -> SET ?

    Where SET is:
        TOKEN
        TOKEN + TOKEN
        TOKEN + TOKEN + TOKEN ...

    Tokens:
        - simple alphanumeric strings (A–Z, a–z, 0–9)
        - optionally must appear in definition["tokens"] if provided

    Raises:
        DslParseError on any violation.
    """
    raw = expr
    expr = expr.strip()

    if not expr:
        raise DslParseError("Empty DSL expression.")

    # basic character check: allow only letters, digits, spaces, '+', '-', '>', '?'
    for ch in expr:
        if ch.isalnum() or ch.isspace() or ch in {"+", "-", ">", "?"}:
            continue
        raise DslParseError(f"Invalid character '{ch}' in DSL expression.")

    # handle question mark
    expr_wo_q, is_question = _strip_question_mark(expr)

    # get token definition map (may be empty)
    tokens_def: Dict[str, str] = definition.get("tokens", {}) if isinstance(definition, dict) else {}

    # check if this is a relation (contains '->')
    if "->" in expr_wo_q:
        # ensure there's exactly one '->'
        if expr_wo_q.count("->") != 1:
            raise DslParseError("Expression may contain at most one '->'.")

        left_str, right_str = [p.strip() for p in expr_wo_q.split("->", 1)]
        if not left_str or not right_str:
            raise DslParseError("Both sides of '->' must be non-empty concept sets.")

        left_tokens = _parse_concept_set(left_str, tokens_def)
        right_tokens = _parse_concept_set(right_str, tokens_def)

        return DslExpression(
            kind="relation",
            left=left_tokens,
            right=right_tokens,
            is_question=is_question,
            raw=raw,
        )

    # otherwise: simple concept set
    left_tokens = _parse_concept_set(expr_wo_q, tokens_def)
    return DslExpression(
        kind="set",
        left=left_tokens,
        right=None,
        is_question=is_question,
        raw=raw,
    )


def is_valid_dsl(expr: str, definition: Dict) -> bool:
    """
    Convenience helper: True if parse_dsl() succeeds, False otherwise.
    """
    try:
        parse_dsl(expr, definition)
        return True
    except DslParseError:
        return False