# Instruction Data Generation Tools

This directory contains tools for generating, validating, and managing instruction training data for the DACH accounting assistant.

## Overview

The instruction data generation pipeline creates high-quality training samples with country-specific handling for Austria (AT), Germany (DE), and Switzerland (CH).

## Tools

### `generate_instruction_data.py`
Main generator for synthetic instruction samples using Azure OpenAI.

**Features:**
- Template-based question generation
- Country-specific answer generation with injected legal rules
- Quality validation before saving
- Metadata tracking (template_id, country, seed, model)

**Usage:**
```bash
python tools/generate_instruction_data.py \
  --templates data/instruction/question_templates.json \
  --country-rules data/instruction/country_rules.json \
  --output data/instruction/synthetic.jsonl \
  --count 100 \
  --model gpt-4
```

**Options:**
- `--templates`: Path to question templates JSON
- `--country-rules`: Path to country rules JSON
- `--output`: Output JSONL file
- `--count`: Number of samples to generate
- `--countries`: Comma-separated country codes (default: AT,DE,CH)
- `--model`: Azure OpenAI deployment name
- `--temperature`: LLM temperature (default: 0.7)
- `--seed`: Random seed for reproducibility
- `--append`: Append to existing output file

**Environment Variables:**
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY`: API key
- `AZURE_OPENAI_DEPLOYMENT`: Deployment name (can be overridden with --model)

### `validate_instruction_data.py`
Automated quality validation for instruction samples.

**Validation Checks:**
- Schema compliance
- Disclaimer presence (where required)
- Legal reference format
- Cautious language usage
- Forbidden absolute statements
- Country consistency

**Usage:**
```bash
python tools/validate_instruction_data.py \
  --input data/instruction/synthetic.jsonl \
  --report validation_report.json \
  --verbose
```

**Options:**
- `--input`: Input JSONL file to validate
- `--report`: Output validation report JSON file
- `--verbose`: Print detailed validation results
- `--flagged-output`: Save flagged samples to separate JSONL file

**Exit Codes:**
- `0`: Validation passed (≥95% pass rate)
- `1`: Validation warning (< 95% pass rate)

### `expand_country_variants.py`
Expands gold instruction samples to AT/DE/CH country-specific variants.

**Features:**
- Adapts legal references for target country
- Adjusts amounts, thresholds, and tax rates
- Maintains original structure and quality
- Uses country rules for accurate adaptation

**Usage:**
```bash
python tools/expand_country_variants.py \
  --input data/instruction/gold.jsonl \
  --country-rules data/instruction/country_rules.json \
  --output data/instruction/gold_expanded.jsonl \
  --countries AT,DE,CH
```

**Options:**
- `--input`: Input JSONL file (typically gold.jsonl)
- `--country-rules`: Path to country rules JSON
- `--output`: Output JSONL file for expanded variants
- `--countries`: Comma-separated country codes (default: AT,DE,CH)
- `--model`: Azure OpenAI deployment name
- `--temperature`: LLM temperature (default: 0.3, lower for more conservative adaptation)
- `--append`: Append to existing output file

**Environment Variables:**
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY`: API key
- `AZURE_OPENAI_DEPLOYMENT`: Deployment name

### `paraphrase_instruction_data.py`
Creates paraphrased variants of instruction samples to increase linguistic diversity.

**Features:**
- Maintains semantic content and structure
- Preserves legal references and disclaimers
- Uses multiple paraphrasing strategies
- Keeps professional tone and cautious language

**Usage:**
```bash
python tools/paraphrase_instruction_data.py \
  --input data/instruction/gold_expanded.jsonl \
  --output data/instruction/paraphrased.jsonl \
  --variants 3 \
  --temperature 0.7
```

**Options:**
- `--input`: Input JSONL file with instruction samples
- `--output`: Output JSONL file for paraphrased samples
- `--variants`: Number of paraphrased variants per sample (default: 2)
- `--model`: Azure OpenAI deployment name
- `--temperature`: LLM temperature (default: 0.7)
- `--seed`: Random seed for reproducibility
- `--sample-fraction`: Fraction of input to paraphrase (0.0-1.0, default: 1.0)
- `--append`: Append to existing output file

**Paraphrasing Strategies:**
- Question reformulation
- Answer reordering (maintaining structure)
- Synonym variation
- Example variation

**Environment Variables:**
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY`: API key
- `AZURE_OPENAI_DEPLOYMENT`: Deployment name

### `split_instruction_data.py`
Splits instruction data into train/val/test sets with stratification and leakage control.

**Features:**
- Stratified splitting by topic, difficulty, or country
- Automatic leakage detection
- Distribution statistics
- Deterministic (seeded) splits

**Usage:**
```bash
python tools/split_instruction_data.py \
  --input data/instruction/all_instruction_data.jsonl \
  --output-dir data/instruction/splits \
  --split 0.8,0.1,0.1 \
  --stratify-by topic \
  --seed 42
```

**Options:**
- `--input`: Input JSONL file with all samples
- `--output-dir`: Output directory for split files
- `--split`: Split ratios as train,val,test (must sum to 1.0, default: 0.8,0.1,0.1)
- `--stratify-by`: Key to stratify by (default: topic)
- `--seed`: Random seed for reproducibility (default: 42)
- `--leakage-keys`: Keys to check for data leakage (default: id,template_id)
- `--no-test-split`: Only create train/val split (no test set)

**Output Files:**
- `train.jsonl`: Training samples
- `val.jsonl`: Validation samples
- `test.jsonl`: Test samples (if not --no-test-split)
- `split_manifest.json`: Metadata and statistics about the split

## Data Files

### `data/instruction/country_rules.json`
Structured knowledge base of country-specific legal differences.

**Example:**
```json
{
  "rule_id": "gwg_grenze",
  "topic": "gwg",
  "description": "Threshold for low-value assets",
  "countries": {
    "AT": {
      "threshold_net": 800,
      "currency": "EUR",
      "legal_reference": "§ 13 EStG",
      "source": "Einkommensteuergesetz Österreich"
    },
    "DE": {...},
    "CH": {...}
  }
}
```

### `data/instruction/question_templates.json`
Reusable templates for generating diverse questions.

**Example:**
```json
{
  "template_id": "tmpl_gwg_001",
  "topic": "gwg",
  "instruction_type": "explanation",
  "difficulty": "basic",
  "country_specific": true,
  "question_templates": [
    "Was gilt als geringwertiges Wirtschaftsgut in {country_name}?"
  ],
  "required_rules": ["gwg_grenze"],
  "answer_structure": {
    "sections": ["kurzantwort", "begruendung", "hinweis"],
    "must_include_legal_ref": true,
    "must_include_disclaimer": true
  }
}
```

## Workflows

### Quick Test Generation
```bash
# Generate 10 samples for testing
python tools/generate_instruction_data.py \
  --count 10 \
  --output data/instruction/test.jsonl

# Validate
python tools/validate_instruction_data.py \
  --input data/instruction/test.jsonl \
  --verbose
```

### Production Generation
```bash
# 1. Generate 500 samples
python tools/generate_instruction_data.py \
  --count 500 \
  --output data/instruction/synthetic.jsonl \
  --model gpt-4 \
  --seed 42

# 2. Validate
python tools/validate_instruction_data.py \
  --input data/instruction/synthetic.jsonl \
  --report validation_report.json \
  --flagged-output flagged_samples.jsonl

# 3. Review flagged samples
# (Manual review of flagged_samples.jsonl)

# 4. Generate more if needed
python tools/generate_instruction_data.py \
  --count 100 \
  --output data/instruction/synthetic.jsonl \
  --append
```

### Complete Pipeline: Gold Samples → Country Variants → Paraphrasing → Train/Val/Test

This workflow shows how to expand the 20 gold samples into a diverse training dataset:

```bash
# Step 1: Expand gold samples to country variants (20 → 60 samples)
python tools/expand_country_variants.py \
  --input data/instruction/gold.jsonl \
  --country-rules data/instruction/country_rules.json \
  --output data/instruction/gold_expanded.jsonl \
  --countries AT,DE,CH \
  --temperature 0.3

# Step 2: Validate expanded samples
python tools/validate_instruction_data.py \
  --input data/instruction/gold_expanded.jsonl \
  --report validation_gold_expanded.json \
  --verbose

# Step 3: Create paraphrased variants (60 → 180 samples with 3 variants each)
python tools/paraphrase_instruction_data.py \
  --input data/instruction/gold_expanded.jsonl \
  --output data/instruction/gold_paraphrased.jsonl \
  --variants 3 \
  --temperature 0.7 \
  --seed 42

# Step 4: Validate paraphrased samples
python tools/validate_instruction_data.py \
  --input data/instruction/gold_paraphrased.jsonl \
  --report validation_paraphrased.json \
  --flagged-output flagged_paraphrased.jsonl

# Step 5: Combine gold expanded + paraphrased samples
cat data/instruction/gold_expanded.jsonl \
    data/instruction/gold_paraphrased.jsonl \
    > data/instruction/all_gold_variants.jsonl

# Step 6: Split into train/val/test
python tools/split_instruction_data.py \
  --input data/instruction/all_gold_variants.jsonl \
  --output-dir data/instruction/splits \
  --split 0.8,0.1,0.1 \
  --stratify-by topic \
  --seed 42

# Result: train.jsonl, val.jsonl, test.jsonl ready for training!
```

**Expected output:**
- Gold: 20 samples
- Expanded (AT/DE/CH): 60 samples
- Paraphrased (3 variants): 180 samples
- Total: 240 samples
- Train: ~192 samples (80%)
- Val: ~24 samples (10%)
- Test: ~24 samples (10%)

### Combining Gold + Synthetic Data

```bash
# 1. Start with expanded gold samples
python tools/expand_country_variants.py \
  --input data/instruction/gold.jsonl \
  --output data/instruction/gold_expanded.jsonl \
  --countries AT,DE,CH

# 2. Generate synthetic samples from templates
python tools/generate_instruction_data.py \
  --count 500 \
  --output data/instruction/synthetic.jsonl \
  --seed 42

# 3. Paraphrase both (to increase diversity)
python tools/paraphrase_instruction_data.py \
  --input data/instruction/gold_expanded.jsonl \
  --output data/instruction/gold_paraphrased.jsonl \
  --variants 2

python tools/paraphrase_instruction_data.py \
  --input data/instruction/synthetic.jsonl \
  --output data/instruction/synthetic_paraphrased.jsonl \
  --variants 1 \
  --sample-fraction 0.5

# 4. Combine all sources
cat data/instruction/gold_expanded.jsonl \
    data/instruction/gold_paraphrased.jsonl \
    data/instruction/synthetic.jsonl \
    data/instruction/synthetic_paraphrased.jsonl \
    > data/instruction/all_combined.jsonl

# 5. Split for training
python tools/split_instruction_data.py \
  --input data/instruction/all_combined.jsonl \
  --output-dir data/instruction/final_splits \
  --split 0.8,0.1,0.1 \
  --stratify-by topic \
  --seed 42
```

## Quality Standards

### Answer Structure
Recommended structure (flexible based on question type):
1. **Kurzantwort** (Short Answer)
2. **Begründung / Einordnung** (Reasoning / Classification)
3. **Voraussetzungen / Prüfungen** (Prerequisites / Checks)
4. **Risiken / Ausnahmen** (Risks / Exceptions)
5. **Hinweis / Disclaimer** (Note / Disclaimer)

### Language Requirements
- **Cautious phrases:** "grundsätzlich", "in der Regel", "abhängig vom Einzelfall"
- **Forbidden:** "immer", "nie", "definitiv"
- **Professional tone:** Sachlich, ruhig, professionell, unterstützend
- **Legal references:** Specific paragraph citations (e.g., "§ 15 Abs. 1 UStG")
- **Disclaimers:** "Diese Darstellung ersetzt keine steuerliche Beratung."

### Validation Thresholds
- **Target pass rate:** ≥98%
- **Warning threshold:** <95%
- **Individual checks:** All should pass for high-quality samples

## Troubleshooting

### Low Validation Pass Rate
**Symptom:** Less than 95% pass rate

**Solutions:**
1. Review validation report for common issues
2. Update generation prompts to emphasize quality requirements
3. Adjust temperature (lower = more conservative)
4. Add post-processing to inject disclaimers

### LLM API Errors
**Symptom:** Rate limits or timeouts

**Solutions:**
1. Check Azure OpenAI logs
2. Add delays between batches
3. Reduce batch size
4. Use retry logic with exponential backoff

### Country-Specific Errors
**Symptom:** Legal references don't match country

**Solutions:**
1. Validate `country_rules.json` has correct references
2. Update generation prompt to explicitly state country context
3. Add country-specific validation in post-processing

## Cost Estimation

**Azure OpenAI Costs (approximate):**
- GPT-4: ~$0.03 per 1K tokens (input), ~$0.06 per 1K tokens (output)
- Average sample: ~500 tokens input, ~800 tokens output
- **Cost per sample:** ~$0.063

**For 500 samples:**
- Generation: ~$31.50

**Cost optimization:**
- Use GPT-3.5 for simpler templates
- Batch processing
- Cache common rule contexts

## References

- **Epic:** `docs/epics_instruction_data.md` - Full implementation plan
- **PRD:** `docs/prd_instruction_data.md` - Product requirements
- **Guide:** `docs/instruction_data_generation.md` - Complete documentation
- **Project Guidelines:** `CLAUDE.md` - Development conventions
