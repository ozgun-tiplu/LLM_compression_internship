# LLM Compression Internship — MedQA DSL Pipeline

This repository contains an experimental pipeline for compressing medical multiple-choice questions (MedQA-style) into a symbolic Domain Specific Language (DSL), maintaining a persistent token definition sheet, and evaluating model performance using the compressed representation.

The objective is to reduce prompt/token cost while preserving enough discriminative medical signal to correctly answer the question.

---

# Core Components

## llm_compression/

### baseline.py
Direct answering baseline (no DSL compression).  
Optionally supports majority voting.

### compress_and_compare.py
Runs:
- Round 1 → Round 2 → Round 4 → Round 5  
(Compression + answer + evaluation, without back-translation)

Computes:
- Accuracy
- Macro-F1

### step1_2_build_token_sheet.py
Step 1 (expensive phase):
- Builds or extends a token sheet from a training dataset.
- Runs R1 → R2 → R4.
- Performs dedupe-by-meaning.
- Rewrites DSL expressions to canonical tokens.
- Outputs:
  - Token sheet (`token_sheet_*.json`)
  - Compressed dataset (`compressed_*.jsonl`)

### step2_map_and_answer_cached_update.py
Step 2 (cheap cached inference):
- Loads existing token sheet.
- Maps new questions using only existing tokens.
- Sends only a minimal glossary for answering.
- Expands sheet only when mapping is too UNK-heavy.
- Outputs:
  - `mapped_and_answered_*.jsonl`

---

## parser/parser.py

Implements strict DSL validation:
- Grammar enforcement
- Token format rules
- Relation parsing
- Question marker rules

Used to guarantee structural consistency of DSL outputs.

---

# DSL Specification

## Tokens
- UPPERCASE alphanumeric
- Length: 2–8 characters
- Example: `AGE33`, `DKA`, `HYPOTN`, `NSCLC`

## Operators
- `+` → combine tokens
- `->` → relation between concept sets
- `?` → question marker (only at end)

## AGE Rule
If the original question contains `"X-year-old"`, DSL must contain:

AGEX

Example:

AGE45 + CHESTPAIN + ST_ELEV ?

---

# Pipeline Rounds

## Round 1 — Initial Compression
Transforms full question into first DSL version.  
Strict token format enforcement.

## Round 2 — Optimization + Sheet Update
- Improves compression.
- Updates token sheet.
- Enforces definition for every token used.

## Round 4 — Consistency Check
- Ensures DSL preserves meaning.
- Fixes minimal drift.
- No back-translation stage.

## Round 5 — Answering
- Uses DSL + token definitions + options.
- Outputs A–E.
- Optional majority voting.

---

# Two-Step Architecture

## Step 1 — Vocabulary Induction (Expensive)
Goal:
- Build strong reusable token inventory.
- Extend sheet incrementally.
- Canonicalize tokens by meaning.

Cost:
- High (full sheet sent to model).

## Step 2 — Cached Mapping (Cheap)
Goal:
- Use existing sheet efficiently.
- Send only:
  - Allowed token names (mapping)
  - Small glossary subset (answering)
- Expand sheet only if UNK ratio threshold exceeded.

Cost:
- Much lower than Step 1.

---

# Input Format

## Questions JSONL
Each line:

```json
{
  "id": 1,
  "question": "...",
  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "...",
    "E": "..."
  }
}
```

## Answers JSONL
Each line:

```json
{
  "id": 1,
  "answer_idx": "C"
}
```

---

# Setup

## Python Version
Tested on:

Python 3.12.7

Recommended:

Python >= 3.10

## Install Dependencies

```
pip install -r requirements.txt
```

## API Key

Create `.env` in repo root:

```
OPENAI_API_KEY=your_key_here
```

---

# Running

## Baseline

```
python -m llm_compression.baseline
```

## Compress + Answer + Evaluate

```
python -m llm_compression.compress_and_compare
```

## Step 1 — Build / Extend Token Sheet

```
python -m llm_compression.step1_2_build_token_sheet
```

## Step 2 — Cached Mapping + Answer

```
python -m llm_compression.step2_map_and_answer_cached_update
```

---

# Design Principles

- Strict DSL grammar enforcement via parser.
- Token deduplication by semantic meaning.
- Atomic writes for sheet persistence.
- Controlled vocabulary growth.
- Cost-aware caching strategy.
- Separation of questions and answers to prevent leakage.

---

# Disclaimer

This is an experimental research prototype for studying compression and structured prompting with LLMs.

Not intended for clinical or diagnostic use.
