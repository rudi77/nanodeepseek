# Epics — Synthetic Training Data (SFT + RL/DPO)

Derived from `docs/prd_synthetic_training_data.md`.

## Epic E0 — Schema + Contracts Freeze

- **Goal**: Define stable dataset + solution schemas (v1) so generation/validation/training are compatible.
- **Scope**:
  - Finalize SFT chat JSONL envelope
  - Finalize DPO JSONL envelope
  - Finalize canonical BookEntry schema v1 (EB-focused) + versioning strategy
- **Deliverables**:
  - Documented schemas + examples
  - Explicit schema version field convention (e.g. `schema_version`)
- **Acceptance criteria**:
  - A sample SFT row and DPO row can be parsed and validated deterministically
  - Schema changes require bumping a version string (no silent breaking changes)

## Epic E1 — Synthetic EB Generator (Templates → SFT + DPO)

- **Goal**: Generate high-volume, numerically correct Eröffnungsbuchung data using rule-based computation + LLM paraphrase.
- **Scope**:
  - Case sampling (template_id, industry, amount model)
  - Rule-based posting computation (Soll/Haben, VAT net→gross, rounding)
  - Instruction paraphrasing via LLM (JSON-only)
  - DPO rejected generation via controlled mutation (single-error when feasible)
- **Deliverables**:
  - `case_library_eb_100.json`
  - `generate_eb_dataset.py` (Azure OpenAI paraphrase + rule-based solver)
  - Output files (baseline MVP):
    - `train_sft_eb_1000.jsonl`
    - `train_dpo_eb_1000.jsonl`
  - Documented required env vars:
    - `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` (+ optional `AZURE_OPENAI_API_VERSION`)
  - Generation config (seed, sizes, quotas)
- **Acceptance criteria**:
  - SFT JSON parse rate ≥ 0.99
  - DPO JSON parse rate ≥ 0.99 (chosen and rejected)
  - Chosen validation pass rate ≥ 0.98 (balance, rounding, VAT math where applicable)
  - Rejected differs from chosen by intended error class in ≥ 0.95 of cases

## Epic E2 — Validation, QA Gates, and Audit Logs

- **Goal**: Prevent low-quality samples from entering datasets; provide traceability and debugging.
- **Scope**:
  - Schema checks (required keys, enums, types)
  - Accounting checks (Soll=Haben totals; VAT consistency; positive amounts; 2 decimals)
  - Rejected checks (must be “wrong” and ideally single-error)
  - Provenance metadata (`source`, `template_id`, `industry`, seeds/config hash)
  - Reject logs with reasons
- **Deliverables**:
  - Validator module usable by all generators
  - Structured reject log output (JSONL/CSV)
  - Summary report (counts, pass/fail reasons)
- **Acceptance criteria**:
  - Generator can run in “validate-only” mode over an existing dataset
  - Any failing record has an actionable error reason and provenance pointer

## Epic E3 — Dataset Splits + Leakage Control

- **Goal**: Produce reliable train/val(/test) splits without accidental leakage that inflates eval.
- **Scope**:
  - Deterministic splitting strategy (seeded)
  - Leakage rules (e.g. avoid same `(template_id, amount_bucket)` across splits)
  - Output separate files (train/val/test)
- **Deliverables**:
  - Splitter utility
  - Split manifests (counts, distributions)
- **Acceptance criteria**:
  - Running twice with same seed yields identical splits
  - Split stats match requested distribution targets (template + industry coverage)

## Epic E4 — Exams Ingestion v1 (Real Q/A + Solutions → SFT + DPO)

- **Goal**: Turn `data/processed/5_pruefungen/*.md` into supervised and preference data.
- **Scope**:
  - Extract question/solution pairs (including tables where possible)
  - Normalize outputs (either structured JSON for accounting tasks or canonical answer text for theory)
  - Optional paraphrase of question text
  - DPO rejected generation by single controlled errors (numeric/concept/bookkeeping)
- **Deliverables**:
  - Extractor script that yields structured items with provenance (file + section)
  - `train_sft_exams_*.jsonl`
  - `train_dpo_exams_*.jsonl`
- **Acceptance criteria**:
  - Provenance pointers allow locating the exact originating section in the markdown
  - DPO rejected is plausibly wrong and fails at least one validator/check in ≥ 0.95 cases (for tasks where a validator exists)

## Epic E5 — Laws Q&A Generator v1 (UGB/UStG/BAO/GoBD → SFT)

- **Goal**: Create robust law-based Q&A style data to improve conceptual/legal language handling.
- **Scope**:
  - Chunking law markdown into sections
  - Q&A generation with bounded style (German, concise, non-hallucinatory)
  - Optional: “citation-aware” answers (store the referenced section heading/snippet as provenance)
- **Deliverables**:
  - `train_sft_laws_*.jsonl` (+ optional DPO)
  - Chunk index with provenance
- **Acceptance criteria**:
  - Each sample contains provenance to a law file + section identifier
  - Output formatting follows the SFT envelope and is parseable ≥ 0.99

## Epic E6 — Mixing Strategy + Training Readiness

- **Goal**: Produce a first “trainable” mixture dataset aligned with your training plan.
- **Scope**:
  - Decide mixing ratios (templates vs exams vs laws)
  - Normalize system prompts and output expectations across sub-datasets
  - Produce a final merged dataset with consistent schema + metadata
- **Deliverables**:
  - `train_sft_all_*.jsonl` and `train_dpo_all_*.jsonl`
  - A single config file describing the build (sizes, ratios, seeds)
- **Acceptance criteria**:
  - Merged dataset passes validation and split checks
  - Training pipeline can consume it without per-source special casing

## Epic E7 — Cost/Throughput Optimizations (Optional)

- **Goal**: Reduce LLM spend and runtime while keeping quality unchanged.
- **Scope**:
  - Batch LLM paraphrase calls (where supported)
  - Cache paraphrases for repeated template+amount patterns (if desired)
  - Rate limiting/backoff and retry strategy
- **Deliverables**:
  - Configurable throttling/caching
  - Benchmark report (cost/sample, samples/min)
- **Acceptance criteria**:
  - Measurable reduction in runtime and/or cost with no drop in validation metrics


