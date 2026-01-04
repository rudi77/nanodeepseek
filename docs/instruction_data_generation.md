# Instruction Data Generation Guide

This document provides a complete guide to generating high-quality instruction training data for the DACH accounting assistant, including country-specific variants for Austria (AT), Germany (DE), and Switzerland (CH).

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Configuration Reference](#configuration-reference)
6. [Quality Standards](#quality-standards)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose
Generate synthetic instruction data that trains the model to:
- Respond with proper structure and cautious language
- Differentiate between country-specific legal frameworks (AT/DE/CH)
- Maintain professional accounting tone
- Include appropriate disclaimers and legal references

### Data Pipeline Flow
```
┌─────────────────────┐
│  Country Rules DB   │ ← Domain expert curated
│  (country_rules.json)│
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Question Templates  │ ← Manually created templates
│ (templates.json)    │
└──────────┬──────────┘
           │
           ├──────────────────┐
           │                  │
┌──────────▼──────────┐  ┌───▼──────────────┐
│  LLM Generation     │  │ Country Variant  │
│  (synthetic.jsonl)  │  │ Expansion        │
└──────────┬──────────┘  └───┬──────────────┘
           │                  │
           └────────┬─────────┘
                    │
           ┌────────▼─────────┐
           │ Validation &     │
           │ Quality Checks   │
           └────────┬─────────┘
                    │
           ┌────────▼─────────┐
           │ Human Review     │
           │ (flagged only)   │
           └────────┬─────────┘
                    │
           ┌────────▼─────────┐
           │ Paraphrasing     │
           │ (diversity)      │
           └────────┬─────────┘
                    │
           ┌────────▼─────────┐
           │ Train/Val/Test   │
           │ Splitting        │
           └──────────────────┘
```

### Key Principles
1. **Country-Specific Correctness**: Legal references and thresholds must match each country
2. **LLM for Language, Not Facts**: Use LLMs for phrasing, inject facts via context
3. **Quality Over Quantity**: Better 500 excellent samples than 5,000 mediocre ones
4. **Audit Trail**: Every sample tracks its generation lineage

---

## Architecture

### Component Overview

#### 1. Country Rules Database (`data/instruction/country_rules.json`)
Structured knowledge base of legal differences across AT/DE/CH.

**Example:**
```json
{
  "topic": "gwg_grenze",
  "description": "Threshold for low-value assets (geringwertige Wirtschaftsgüter)",
  "countries": {
    "AT": {
      "threshold_net": 800,
      "currency": "EUR",
      "legal_reference": "§ 13 EStG (AT)",
      "source": "Einkommensteuergesetz Österreich"
    },
    "DE": {
      "threshold_net": 800,
      "currency": "EUR",
      "legal_reference": "§ 6 Abs. 2 EStG (DE)",
      "source": "Einkommensteuergesetz Deutschland"
    },
    "CH": {
      "threshold_net": 1000,
      "currency": "CHF",
      "legal_reference": "Art. 29 DBG",
      "source": "Bundesgesetz über die direkte Bundessteuer"
    }
  },
  "common_across_dach": false
}
```

#### 2. Question Templates (`data/instruction/question_templates.json`)
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
    "Was gilt als geringwertiges Wirtschaftsgut in {country_name}?",
    "Bis zu welchem Betrag kann ein Wirtschaftsgut in {country_name} sofort abgeschrieben werden?",
    "Welche Grenze gilt für GWG in {country_name}?"
  ],
  "required_rules": ["gwg_grenze"],
  "answer_structure": {
    "sections": ["kurzantwort", "begruendung", "einschraenkungen", "hinweis"],
    "must_include_legal_ref": true,
    "must_include_disclaimer": true
  }
}
```

#### 3. Generator Scripts

##### `tools/generate_instruction_data.py`
Main synthetic data generator using Azure OpenAI.

**Key Features:**
- Template-based question generation
- Country-specific answer generation with injected rules
- Quality validation before saving
- Metadata tracking (template_id, country, seed, model)

##### `tools/expand_country_variants.py`
Creates AT/DE/CH variants of existing samples.

**Key Features:**
- Analyzes topic for country-specific rules
- Generates 3 variants with adapted legal references
- Validates factual correctness per country

##### `tools/validate_instruction_data.py`
Automated quality validation.

**Checks:**
- Schema compliance
- Disclaimer presence
- Legal reference format
- Cautious language phrases
- No absolute statements
- Country consistency

##### `tools/review_samples.py`
Human review interface for flagged samples.

**Features:**
- Display flagged samples with validation issues
- Accept/reject with notes
- Batch review support
- Statistics dashboard

##### `tools/paraphrase_instruction_data.py`
Diversity expansion through paraphrasing.

**Features:**
- Generate 3-5 paraphrases per sample
- Preserve structure and legal references
- Semantic similarity validation
- Spot-check human review

##### `tools/split_instruction_data.py`
Train/val/test splitting with stratification.

**Features:**
- Stratify by topic/country/difficulty
- Oversample gold samples in training
- Generate coverage reports
- Configurable split ratios

---

## Prerequisites

### Environment Setup

#### 1. Python Environment
```bash
# Ensure uv is installed
uv --version

# Install dependencies (from project root)
uv pip install openai jsonschema pandas numpy scikit-learn
```

#### 2. Azure OpenAI Configuration
Set environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"  # e.g., gpt-4
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"  # optional
```

Or create `.env` file:
```bash
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

#### 3. Domain Expert Access
For validating `country_rules.json` and reviewing generated samples.

---

## Step-by-Step Guide

### Phase 1: Foundation (Stories 1-2)

#### Step 1: Create Country Rules Database
```bash
# Create country rules with domain expert
# File: data/instruction/country_rules.json

# Example topics to cover:
# - VAT rates and thresholds
# - Small business exemptions
# - GWG limits
# - Invoice requirements
# - Travel expense rules
# - Entertainment deduction limits
```

**Validation:**
```bash
# Validate JSON structure
python -c "import json; json.load(open('data/instruction/country_rules.json'))"

# Domain expert review checklist:
# - [ ] Legal references are correct
# - [ ] Thresholds are current (2024/2025)
# - [ ] Source documents are cited
# - [ ] All three countries (AT/DE/CH) covered
```

#### Step 2: Create Question Templates
```bash
# Create templates for each major topic
# File: data/instruction/question_templates.json

# Recommended: 10-20 templates per topic
# - ust (VAT)
# - abschreibung (depreciation)
# - belege (documents)
# - gwg (low-value assets)
# - reisekosten (travel expenses)
# - bewirtung (entertainment)
```

### Phase 2: Synthetic Generation (Story 3)

#### Step 3: Generate Synthetic Samples
```bash
# Generate 500 synthetic samples
python tools/generate_instruction_data.py \
  --templates data/instruction/question_templates.json \
  --country-rules data/instruction/country_rules.json \
  --output data/instruction/synthetic.jsonl \
  --count 500 \
  --model gpt-4 \
  --seed 42

# Monitor progress
tail -f logs/generation.log
```

**Expected Output:**
```
Generating instruction samples...
Progress: 100/500 (20.0%) | Validated: 98/100 (98.0%)
Progress: 200/500 (40.0%) | Validated: 196/200 (98.0%)
...
Complete: 500/500 (100.0%) | Validated: 489/500 (97.8%)

Saved to: data/instruction/synthetic.jsonl
Flagged for review: 11 samples (see logs/flagged.jsonl)
```

### Phase 3: Country Variants (Story 4)

#### Step 4: Generate Country-Specific Variants
```bash
# Expand gold samples to AT/DE/CH variants
python tools/expand_country_variants.py \
  --input data/instruction/gold.jsonl \
  --country-rules data/instruction/country_rules.json \
  --output data/instruction/gold_country_variants.jsonl \
  --countries AT,DE,CH

# Expected: 20 gold samples → ~60 country-specific variants
```

### Phase 4: Quality Assurance (Story 5)

#### Step 5: Validate Generated Samples
```bash
# Run automated validation
python tools/validate_instruction_data.py \
  --input data/instruction/synthetic.jsonl \
  --report validation_report.json \
  --verbose

# Review report
cat validation_report.json
```

**Example Report:**
```json
{
  "total_samples": 500,
  "validation_results": {
    "schema_valid": 500,
    "has_disclaimer": 487,
    "has_legal_ref": 452,
    "cautious_language": 478,
    "no_absolutes": 489,
    "country_consistency": 495
  },
  "pass_rate": 0.978,
  "flagged_for_review": 11,
  "common_issues": [
    {"issue": "missing_disclaimer", "count": 13},
    {"issue": "weak_legal_ref", "count": 48}
  ]
}
```

#### Step 6: Human Review of Flagged Samples
```bash
# Review flagged samples
python tools/review_samples.py \
  --input data/instruction/synthetic.jsonl \
  --flagged-only \
  --interactive

# Approve, reject, or edit samples
```

### Phase 5: Diversity Expansion (Story 6)

#### Step 7: Generate Paraphrases
```bash
# Create 3 paraphrases per approved sample
python tools/paraphrase_instruction_data.py \
  --input data/instruction/synthetic.jsonl \
  --output data/instruction/paraphrased.jsonl \
  --variants 3 \
  --model gpt-4 \
  --only-approved

# This expands 500 samples → 1,500 paraphrased variants
```

### Phase 6: Dataset Preparation (Story 7)

#### Step 8: Create Train/Val/Test Splits
```bash
# Combine all sources and split
python tools/split_instruction_data.py \
  --inputs data/instruction/gold.jsonl \
           data/instruction/gold_country_variants.jsonl \
           data/instruction/synthetic.jsonl \
           data/instruction/paraphrased.jsonl \
  --output-dir data/instruction/splits/ \
  --train-ratio 0.85 \
  --val-ratio 0.10 \
  --test-ratio 0.05 \
  --stratify topic,country,difficulty \
  --gold-weight 5.0

# Output files:
# - data/instruction/splits/train.jsonl
# - data/instruction/splits/val.jsonl
# - data/instruction/splits/test.jsonl
# - data/instruction/splits/coverage_report.txt
```

**Example Coverage Report:**
```
Dataset Statistics
==================
Total Samples: 1,847

By Split:
- Train: 1,570 (85.0%) - includes oversampled gold
- Val:     185 (10.0%)
- Test:     92 (5.0%)

By Topic:
- ust:            587 (31.8%)
- abschreibung:   324 (17.5%)
- belege:         289 (15.6%)
- gwg:            195 (10.6%)
- reisekosten:    178 (9.6%)
- bewirtung:      145 (7.8%)
- allgemein:      129 (7.0%)

By Country:
- AT:     615 (33.3%)
- DE:     617 (33.4%)
- CH:     615 (33.3%)

By Difficulty:
- basic:        738 (40.0%)
- intermediate: 831 (45.0%)
- advanced:     278 (15.0%)

By Source:
- gold:         20 (1.1%) - oversampled 5× in train
- gold_variants: 60 (3.2%)
- synthetic:   489 (26.5%)
- paraphrased: 1278 (69.2%)
```

---

## Configuration Reference

### Generator Configuration (`config/instruction_generation.yaml`)

```yaml
# LLM Settings
llm:
  provider: azure_openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 1500

# Generation Settings
generation:
  target_count: 500
  batch_size: 10
  retry_on_error: 3
  validation_on_generate: true

# Country Settings
countries:
  enabled: [AT, DE, CH]
  generate_variants: true

# Quality Settings
quality:
  min_validation_score: 0.95
  require_human_review_if_score_below: 0.90
  auto_reject_if_score_below: 0.70

# Output Settings
output:
  format: jsonl
  include_metadata: true
  save_intermediate: true
  log_level: INFO
```

### Validation Rules Configuration

```python
# In tools/validate_instruction_data.py

VALIDATION_RULES = {
    "required_fields": ["id", "type", "source", "topic", "language", "messages", "meta"],
    "required_message_roles": ["user", "assistant"],
    "required_meta_fields": ["difficulty", "contains_legal_reference", "reviewed"],

    "cautious_phrases": [
        "grundsätzlich",
        "in der Regel",
        "abhängig vom Einzelfall",
        "unter bestimmten Voraussetzungen",
        "kann",
        "sollte"
    ],

    "forbidden_phrases": [
        "immer",
        "nie",
        "definitiv",
        "auf jeden Fall",
        "garantiert"
    ],

    "required_disclaimer_phrases": [
        "steuerliche Beratung",
        "Einzelfallprüfung",
        "ersetzt keine"
    ],

    "legal_reference_patterns": [
        r"§\s*\d+",  # § 15
        r"Art\.\s*\d+",  # Art. 29
        r"Abs\.\s*\d+",  # Abs. 1
        r"Nr\.\s*\d+"   # Nr. 2
    ]
}
```

---

## Quality Standards

### Answer Structure Template

Every assistant response should follow this structure (flexible based on question type):

```
1. Kurzantwort (Short Answer)
   - Direct answer to the question
   - 1-2 sentences

2. Begründung / Einordnung (Reasoning / Classification)
   - Legal basis or accounting principle
   - Include specific legal reference (e.g., "§ 15 Abs. 1 UStG")

3. Voraussetzungen / Prüfungen (Prerequisites / Checks)
   - Conditions that must be met
   - What to verify

4. Risiken / Ausnahmen (Risks / Exceptions)
   - Edge cases
   - Common pitfalls
   - Exclusions

5. Hinweis / Disclaimer
   - Professional disclaimer
   - "Diese Darstellung ersetzt keine steuerliche Beratung."
```

### Tone & Language Requirements

**Required Characteristics:**
- ✅ Sachlich (factual, objective)
- ✅ Ruhig (calm, measured)
- ✅ Professionell (professional)
- ✅ Unterstützend (supportive, helpful)
- ✅ Vorsichtig (cautious, qualified statements)

**Forbidden:**
- ❌ Umgangssprache (colloquial language)
- ❌ Marketing-Sprech (marketing language)
- ❌ Absolute statements without qualification
- ❌ Overly technical jargon without explanation

### Country-Specific Validation

**For each country variant:**
1. Legal references match country's legal system
   - AT: UStG (AT), EStG (AT), UGB
   - DE: UStG (DE), EStG (DE), HGB
   - CH: MWSTG, DBG, OR

2. Currency correct
   - AT/DE: EUR (€)
   - CH: CHF

3. Thresholds accurate for country and year

4. Language appropriate
   - AT: Austrian German terms where applicable
   - DE: German German
   - CH: Swiss German terms where applicable

---

## Troubleshooting

### Issue: Low Validation Pass Rate

**Symptom:** Less than 95% of generated samples pass validation

**Diagnosis:**
```bash
# Check validation report for common issues
python tools/validate_instruction_data.py --input synthetic.jsonl --verbose

# Common issues:
# - Missing disclaimers
# - Weak legal references
# - Absolute language
```

**Solution:**
1. Update generation prompts to emphasize quality requirements
2. Inject more explicit examples in system prompt
3. Lower temperature for more conservative outputs
4. Add post-processing step to inject disclaimers

### Issue: Country-Specific Errors

**Symptom:** Legal references don't match country

**Diagnosis:**
```bash
# Check country consistency
grep -E "country.*AT" synthetic.jsonl | grep "EStG (DE)"
```

**Solution:**
1. Validate `country_rules.json` has correct references
2. Update generation prompt to explicitly state country context
3. Add country-specific validation in post-processing

### Issue: LLM API Errors

**Symptom:** Rate limits or timeouts

**Diagnosis:**
```bash
# Check logs
tail -f logs/generation.log | grep ERROR
```

**Solution:**
```python
# In generate_instruction_data.py, add retry logic:
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
def call_llm(prompt):
    ...

# Or add batch delays:
time.sleep(1)  # Between batches
```

### Issue: Poor Paraphrase Quality

**Symptom:** Paraphrases are too similar or change meaning

**Diagnosis:**
```bash
# Run semantic similarity check
python tools/check_paraphrase_quality.py --input paraphrased.jsonl
```

**Solution:**
1. Increase temperature for paraphrasing (e.g., 0.8-0.9)
2. Update paraphrase prompt to encourage more variation
3. Add human spot-check review (10% sample)

---

## Example Workflows

### Workflow 1: Quick 100-Sample Generation
```bash
# Fast iteration for testing
python tools/generate_instruction_data.py \
  --templates data/instruction/question_templates.json \
  --country-rules data/instruction/country_rules.json \
  --count 100 \
  --model gpt-4 \
  --output data/instruction/test_batch.jsonl

python tools/validate_instruction_data.py \
  --input data/instruction/test_batch.jsonl

# Review quality, iterate on prompts
```

### Workflow 2: Full Production Pipeline
```bash
# 1. Generate 500 synthetic samples
python tools/generate_instruction_data.py --count 500

# 2. Generate country variants
python tools/expand_country_variants.py --input gold.jsonl

# 3. Validate all
python tools/validate_instruction_data.py --input synthetic.jsonl

# 4. Human review
python tools/review_samples.py --flagged-only

# 5. Paraphrase approved samples
python tools/paraphrase_instruction_data.py --only-approved --variants 3

# 6. Split into train/val/test
python tools/split_instruction_data.py --gold-weight 5.0
```

### Workflow 3: Incremental Updates
```bash
# Add 50 more samples to existing dataset
python tools/generate_instruction_data.py \
  --count 50 \
  --append-to data/instruction/synthetic.jsonl \
  --start-id synthetic_501

# Validate new samples only
python tools/validate_instruction_data.py \
  --input data/instruction/synthetic.jsonl \
  --only-new

# Re-split with updated data
python tools/split_instruction_data.py --incremental
```

---

## References

- **Epic:** `docs/epics_instruction_data.md` - Full epic breakdown
- **PRD:** `docs/prd_instruction_data.md` - Product requirements
- **Schemas:** `docs/schemas_synthetic_training_data.md` - Data schemas
- **Project Guidelines:** `CLAUDE.md` - Development conventions

---

## Appendix

### A. Cost Estimation

**Azure OpenAI Costs (approximate):**
- GPT-4: ~$0.03 per 1K tokens (input), ~$0.06 per 1K tokens (output)
- Average sample: ~500 tokens input, ~800 tokens output

**Cost per sample:**
- Generation: ~$0.063
- Paraphrasing (3×): ~$0.189
- Total per base sample: ~$0.25

**For 500 base samples:**
- Generation: ~$31.50
- Paraphrasing: ~$94.50
- **Total: ~$126**

**Cost optimization:**
- Use GPT-3.5 for paraphrasing: ~$0.002/sample → Total: ~$35
- Use smaller batches for testing

### B. Sample Metadata Schema

```json
{
  "generation_metadata": {
    "template_id": "tmpl_gwg_001",
    "country": "AT",
    "model": "gpt-4",
    "temperature": 0.7,
    "generated_at": "2024-01-15T10:30:00Z",
    "seed": 42,
    "version": "1.0"
  },
  "validation_metadata": {
    "validated": true,
    "validation_score": 0.98,
    "validation_issues": [],
    "validated_at": "2024-01-15T10:31:00Z"
  },
  "review_metadata": {
    "reviewed": true,
    "reviewer": "domain_expert_1",
    "approved": true,
    "review_notes": "",
    "reviewed_at": "2024-01-16T09:00:00Z"
  },
  "lineage": {
    "derived_from": "gold_003",
    "derivation_type": "country_variant | paraphrase | template",
    "paraphrase_variant": 2
  }
}
```
