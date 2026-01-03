# PRD — Synthetic Training Data Generation (SFT + RL/DPO) for NanoDeepSeek

## 1) Summary

Build a **repeatable data-generation pipeline** that creates high-quality **German (DACH/AT) accounting** training data for:

- **SFT** (supervised fine-tuning) in **chat JSONL** format
- **RL preference training** via **DPO-style** prompt/chosen/rejected pairs (JSONL)

The core principle (from `docs/data_generation.md`) is:

- **Correctness is computed rule-based** (accounting math + booking logic)
- The LLM is used only for **natural-language paraphrasing** (task/instruction text), not for the bookkeeping solution

In addition, leverage `data/processed` (laws + exams with solutions) to create **realistic** instructions, explanations, and edge-case coverage.

## 2) Problem & Motivation

Training a small domain model (or a domain-adapted model) to produce correct accounting outputs is bottlenecked by:

- **Format robustness** (JSON-only outputs, stable schemas)
- **Numerical correctness** (VAT/net/gross, rounding, Soll/Haben balance)
- **Coverage breadth** (industries, account types, typical Austrian/DACH bookkeeping cases)
- **Preference signals** (plausible-but-wrong outputs to teach “don’t do this”)

Manual labeling is expensive; naïve LLM generation hallucinates numbers and accounts. This PRD defines a pipeline to generate scalable data while preserving correctness.

## 3) Goals (What success looks like)

- **G1 — Deterministic correctness**
  - For each synthetic case, a rule-based solver produces the canonical, correct booking output.
  - Validation catches and rejects inconsistent samples (Soll=Haben, VAT math, rounding, schema).

- **G2 — High parse rate**
  - SFT dataset: **>99%** JSON parse success for assistant messages.
  - DPO dataset: **>99%** parse success for both chosen and rejected strings.

- **G3 — Useful negative samples**
  - Rejected samples are **plausible** and contain **exactly one** controlled error class (when feasible).

- **G4 — Breadth**
  - Coverage across industries and typical accounting areas (starting with Eröffnungsbuchungen, extendable).

- **G5 — Reproducible**
  - Same seed/config → deterministic case distribution and stable outputs (except for paraphrase text variation).

## 4) Non-goals (Explicitly out of scope for this PRD)

- Building an end-user product UI or accounting assistant application.
- Implementing a full RAG system over laws/exams.
- Perfect legal interpretation of every edge case in UGB/UStG (we use laws/exams primarily as text sources and prompts, not as the source of numeric truth).
- Multi-country tax engines (initial focus: **AT/DACH-style**; CH content may remain as reading/extraction material).

## 5) Users & Stakeholders

- **Primary user**: You (model trainer) generating datasets for SFT and RL(DPO).
- **Secondary user**: Future evaluation/benchmark scripts consuming datasets for regression tests.
- **Stakeholders**:
  - Model training pipeline (Axolotl/TRL or your own trainer)
  - Dataset curation/quality (validation rules and audits)

## 6) Inputs / Data Sources

### 6.1 Template library (synthetic truth source)

Initial seed described in `docs/data_generation.md`:

- **Case library**: 100 Eröffnungsbuchung templates (EB-001 … EB-100)
- Each template includes: `template_id`, `category`, `sub_category`, `industry_focus`, `description`, `amount_model`, `booking`, `rules`

#### 6.1.1 Concrete MVP package (files + environment)

The MVP is concretely defined by the following artifacts (as provided in `docs/data_generation.md`):

- **Template file**: `case_library_eb_100.json`
- **Generator script**: `generate_eb_dataset.py`
- **Outputs**:
  - `train_sft_eb_1000.jsonl`
  - `train_dpo_eb_1000.jsonl`
- **Required environment variables**:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT`
  - optional: `AZURE_OPENAI_API_VERSION` (defaults in script)

The intended execution is:

- Save JSON as `case_library_eb_100.json`
- Save script as `generate_eb_dataset.py`
- Run: `python generate_eb_dataset.py`

### 6.2 Curated “real” text sources (grounding + realism)

Located in `data/processed/`:

- **Laws** (`data/processed/1_fachtexte_regelwerke/*.md`)
  - Examples: UGB, UStG 1994, BAO, GoBD change note, EStG
  - Use cases:
    - Generate definition/knowledge Q&A (SFT)
    - Generate “explain the rule / cite the basis” style instructions
    - Create edge-case prompts for robustness (but not numeric truth)

- **Exams with solutions** (`data/processed/5_pruefungen/*.md`)
  - Use cases:
    - Extract question → solution pairs (SFT)
    - Create paraphrased variants (SFT)
    - Build preference pairs (DPO) by introducing controlled single errors into the provided solution

## 7) Output Datasets & Schemas

### 7.1 SFT (chat JSONL)

One JSON object per line:

- **Required**: `messages: [{role, content}, ...]`
- Recommended:
  - `messages[0]` system: instruct JSON-only, accounting context
  - `messages[1]` user: natural language task (paraphrase)
  - `messages[2]` assistant: canonical booking JSON (stringified)
- Optional:
  - `meta`: template_id, industry, error-free flags, source (`synthetic_template` vs `processed_exam` vs `processed_law`)

### 7.2 DPO / preference dataset (JSONL)

One JSON object per line:

- `prompt`: instruction text
- `chosen`: correct solution JSON (stringified)
- `rejected`: plausible wrong solution JSON (stringified)
- `meta`: template_id/source, error_class, industry, seed

### 7.3 Canonical “booking output” schema (v1)

Initial schema (from `docs/data_generation.md` sample) for EB:

- `datum` (string)
- `industry` (string)
- `template_id` (string)
- `text` (string)
- `lines`: array of 2 lines:
  - `account_label` (string)
  - `side` ("Soll"|"Haben")
  - `amount` (number, 2 decimals)
  - optional `ekr_code` (string)

**Requirement**: all downstream generators must commit to a stable schema version (e.g. add `"schema_version": "bookentry.v1"` when you move beyond EB).

## 8) Core Workflow (End-to-end)

### 8.1 Synthetic EB pipeline (template → SFT/DPO)

This pipeline is implemented by the reference logic in `generate_eb_dataset.py` (as shown in `docs/data_generation.md`):

1. **Sample a template** (respect `industry_focus`)
2. **Sample an amount** via `pick_amount()`
   - uses a **log-ish distribution** within `min/max`
   - returns `(amount_display, net_amount, vat_rate)` where `amount_display` is what the instruction shows
3. **Compute correct posting rule-based** via `compute_posting()`
   - Rounding to 2 decimals via `round2()`
   - VAT logic: `rules.vat_handling == "net_to_gross"` computes **gross** and creates an instruction hint like `Netto …, USt …% -> brutto buchen.`
   - Creates two booking lines: **Soll** and **Haben**
   - Optionally attaches `ekr_code` if `account_label` is present in `EKR_MAP`
4. **Generate instruction text via LLM paraphrase** via `paraphrase_instruction()`
   - System prompt requires JSON-only output: `{"instruction":"..."}`
   - User prompt injects: industry, date, description, amount, hint
5. **Write SFT row** via `make_sft_chat()`
   - `messages = [system, user, assistant]` where assistant is the JSON booking object (string)
6. **Create rejected variant (rule-based)** via `mutate_rejected()`
   - swap Soll/Haben, or perturb amount, or change one account label (with optional EKR adjustment)
7. **Write DPO row**
   - `{prompt: instruction, chosen: <json string>, rejected: <json string>, meta: {...}}`

### 8.2 “Real” exams pipeline (question/solution → SFT/DPO)

1. **Extract structured pairs** from markdown:
   - question block(s)
   - solution block(s) (tables, formulas, text)
2. **Normalize solutions** into a machine-checkable form where possible:
   - For accounting journal entries: canonical JSON schema
   - For theory Q&A: canonical answer text, optionally with citations
3. **Paraphrase** questions (optional), keep solutions fixed
4. **Generate DPO rejected** by introducing a single controlled error:
   - numeric error (rounding, swapped operands)
   - concept error (wrong classification FIBU/BEBU)
   - bookkeeping error (wrong side/account)

### 8.3 Laws pipeline (text → knowledge Q&A + robustness prompts)

1. Chunk law markdown into sections
2. Generate:
   - definition Q&A (e.g., “Was ist ein Unternehmer nach UGB?”)
   - compliance/GoBD Q&A (esp. for DE)
   - VAT conceptual Q&A (UStG)
3. Optional DPO:
   - rejected answers that are *almost* correct but violate a key term/condition

## 9) Functional Requirements

- **FR1 — Configurable generation**
  - Total samples, per-template quotas, per-source quotas, random seed, industry distribution.

- **FR2 — Strict output formats**
  - LLM paraphrase outputs JSON-only.
  - SFT/DPO lines are valid JSON per line.

- **FR3 — Deterministic solver**
  - Given template + amount + rules → identical booking output.

- **FR4 — Validation gate**
  - Schema validation (required keys, allowed enums)
  - Accounting checks:
    - totals balance (sum Soll == sum Haben)
    - VAT math when applicable
    - amounts positive, 2 decimals

- **FR5 — Rejected generation**
  - Controlled set of error classes
  - Prefer “single-error” mutations

- **FR6 — Dataset splits**
  - Produce train/val/test (or train/val) with stable sampling and no leakage (e.g. template_id+amount buckets).

- **FR7 — Provenance metadata**
  - Every sample includes `source` and identifiers enabling audit/replay.

## 10) Non-functional Requirements

- **NFR1 — Reproducibility**
  - Seedable randomness; store config used for generation.

- **NFR2 — Cost control**
  - LLM calls only for paraphrase (short outputs); batch where possible; rate-limit friendly.

- **NFR3 — Maintainability**
  - Minimal dependencies; use existing Python packaging (project already uses `uv`).

- **NFR4 — Safety**
  - Avoid generating private personal data; use synthetic company names.

## 11) Quality Metrics & Acceptance Criteria

- **Parse rate**
  - SFT assistant content JSON parse: ≥ 0.99
  - DPO chosen/rejected JSON parse: ≥ 0.99

- **Validation pass rate (chosen)**
  - ≥ 0.98 (remaining failures should be logged with reason)

- **Rejected “wrongness”**
  - ≥ 0.95 of rejected samples fail at least one validation rule (or differ from chosen by intended error class)

- **Coverage**
  - Each EB template appears at least N times (configurable)
  - Industry distribution meets target (uniform or weighted)

## 12) Data Governance / Legal Notes

- `data/processed` includes **laws** and **exam materials**. Treat these as:
  - internal training inputs
  - with provenance tracking
  - and explicit dataset source labeling

This PRD does not decide redistribution policy. If you plan to publish the dataset, add a review step for licensing/rights per source document.

## 13) Rollout Plan (Milestones)

- **M0 — PRD + schema freeze**
  - Confirm canonical schemas for EB BookEntry v1 and DPO v1.

- **M1 — EB synthetic v1**
  - Generate 1k SFT + 1k DPO from EB templates (as in `docs/data_generation.md`)
  - Implement validation and logs

- **M2 — Scale + distribution control**
  - Quotas per template/industry; train/val split; reproducible configs

- **M3 — Exams ingestion v1**
  - Extract at least one exam file into structured Q/A and/or bookkeeping tasks + solutions
  - Add paraphrase + DPO mutation for those tasks

- **M4 — Laws Q&A v1**
  - Generate law-based Q&A dataset and integrate into SFT mix

## 14) Open Questions (need your decisions)

- **OQ1 — Target account system**: keep labels + optional `ekr_code`, or enforce EKR codes everywhere?
- **OQ2 — Scope expansion**: beyond Eröffnungsbuchungen, which modules next (ER/AR invoices, Skonto, Reverse-Charge, Anlagen/AfA)?
- **OQ3 — Output schema**: keep the EB-style `lines` schema, or move to a unified journal-entry schema for all tasks?
- **OQ4 — Mixing ratios**: what % of SFT should be (templates vs exams vs laws) for your first training run?


