# Synthetic Data Tools

This folder contains generators and QA utilities for the synthetic training data epics.

## EB synthetic generator

- Script: `tools/generate_eb_dataset.py` (wrapper for `tools/eb_generator.py`)
- Case library: `tools/case_library_eb_100.json`
- Outputs: `train_sft_eb_1000.jsonl`, `train_dpo_eb_1000.jsonl`

Required env vars for Azure OpenAI paraphrase (optional, fallback available):

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- optional: `AZURE_OPENAI_API_VERSION`

Example:

```
python tools/generate_eb_dataset.py --sft-count 1000 --dpo-count 1000
```

## Tool-Use synthetic generator

- Script: `tools/generate_tool_use_dataset.py` (wrapper for `tools/tool_use_generator.py`)
- Template library: `tools/case_library_tool_use_100.json`
- Tool definitions: `tools/tool_definitions.json`
- Outputs: `train_sft_tool_use_1000.jsonl`, `train_dpo_tool_use_1000.jsonl`

**Features:**
- Rule-based tool implementations (calculator, account_lookup, date_calculator, vat_lookup, balance_checker)
- Multi-turn conversations with tool calls
- Optional Azure OpenAI paraphrase (fallback: template text)
- DPO pairs with wrong tool calls (WRONG_TOOL, WRONG_PARAM, MISSING_PARAM)

Example:

```
# Without LLM (rule-based only)
python tools/generate_tool_use_dataset.py --sft-count 1000 --dpo-count 1000 --no-paraphrase

# With Azure OpenAI paraphrase
python tools/generate_tool_use_dataset.py --sft-count 1000 --dpo-count 1000
```

See: `docs/tool_use_training_data.md` for detailed documentation.

## Validation and QA

- Validate: `tools/validate_dataset.py`
- Split: `tools/split_dataset.py`

Example:

```
python tools/validate_dataset.py --sft train_sft_eb_1000.jsonl --dpo train_dpo_eb_1000.jsonl
python tools/split_dataset.py --input train_sft_eb_1000.jsonl --out-train train.jsonl --out-val val.jsonl --out-test test.jsonl
```

## Exams and laws ingestion

- Exams: `tools/exams_ingest.py`
- Laws: `tools/laws_qa_generator.py`

Example:

```
python tools/exams_ingest.py --input data/processed/5_pruefungen/*.md
python tools/laws_qa_generator.py --input data/processed/1_fachtexte_regelwerke/*.md
```

## Mixing

- Mix config: `tools/mix_config.json`
- Script: `tools/mix_datasets.py`

Example:

```
python tools/mix_datasets.py --config tools/mix_config.json
```
