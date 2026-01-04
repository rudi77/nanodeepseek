# Epic: Instruction Data Generation for DACH Accounting Assistant

## Epic Overview

**Goal:** Build a scalable, LLM-powered pipeline to generate high-quality instruction training data for the DACH accounting assistant, with proper handling of country-specific legal and procedural differences across Austria (AT), Germany (DE), and Switzerland (CH).

**Context:** The 20 gold samples in `data/instruction/gold.jsonl` provide a quality baseline. This epic extends coverage through semi-synthetic expansion while maintaining factual correctness and professional tone.

**Success Metrics:**
- Generate 500-2,000 high-quality instruction samples
- Achieve ≥95% human review approval rate
- Cover all major topics with country-specific variants where applicable
- Maintain schema compliance and quality standards from PRD

---

## Story 1: Country-Specific Knowledge Base

**As a** data engineer
**I want** a structured knowledge base of country-specific accounting rules
**So that** generated instruction data reflects correct legal frameworks for AT/DE/CH

### Acceptance Criteria
- [ ] Create `data/instruction/country_rules.json` with structured differences
- [ ] Cover key topics: VAT rates, thresholds, depreciation rules, GWG limits
- [ ] Include legal references (§ paragraphs) per country
- [ ] Document source for each rule (law, regulation, official guidance)
- [ ] Validate with domain expert review

### Technical Details
**Schema:**
```json
{
  "topic": "ust_vorsteuerabzug | gwg_grenze | kleinunternehmer | ...",
  "countries": {
    "AT": {
      "rule": "Description of Austrian rule",
      "legal_reference": "§ X UStG (AT)",
      "thresholds": {...},
      "notes": "Special considerations"
    },
    "DE": {...},
    "CH": {...}
  },
  "common_across_dach": false,
  "source": "UStG, EStG, etc."
}
```

### Example Topics to Cover
- VAT rates (standard, reduced)
- Small business thresholds (Kleinunternehmer)
- GWG limits and depreciation rules
- Invoice requirements
- Travel expense rules
- Entertainment deduction limits
- Cross-border transactions

**Estimated Effort:** 3-5 days
**Dependencies:** None
**Output:** `data/instruction/country_rules.json`

---

## Story 2: Topic Template Library

**As a** data engineer
**I want** a library of question templates by topic and country
**So that** synthetic data covers realistic user questions systematically

### Acceptance Criteria
- [ ] Create `data/instruction/question_templates.json`
- [ ] Define 10-20 templates per major topic
- [ ] Include placeholders for country-specific values
- [ ] Mark which templates require country-specific answers
- [ ] Categorize by difficulty level and instruction type

### Technical Details
**Schema:**
```json
{
  "template_id": "tmpl_ust_001",
  "topic": "ust",
  "instruction_type": "explanation | case_assessment | checklist | process | uncertainty",
  "difficulty": "basic | intermediate | advanced",
  "country_specific": true,
  "question_templates": [
    "Wann ist der Vorsteuerabzug in {country_name} grundsätzlich zulässig?",
    "Welche Umsatzgrenze gilt für die Kleinunternehmerregelung in {country_name}?"
  ],
  "required_country_rules": ["ust_vorsteuerabzug", "kleinunternehmer_grenze"],
  "answer_structure": "kurzantwort_begruendung_hinweis"
}
```

### Template Categories
1. **Explanation Templates** (Was ist...?, Wie funktioniert...?)
2. **Case Assessment Templates** (Wie behandle ich...?, Darf ich...?)
3. **Checklist Templates** (Welche Pflichtangaben...?, Was muss ich prüfen...?)
4. **Process Templates** (Wie gehe ich vor...?, Welche Schritte...?)
5. **Uncertainty Templates** (Kommt darauf an cases, mixed-use scenarios)

**Estimated Effort:** 2-3 days
**Dependencies:** Story 1 (country rules)
**Output:** `data/instruction/question_templates.json`

---

## Story 3: LLM-Based Synthetic Expansion Pipeline

**As a** data engineer
**I want** a script that uses LLMs to expand gold samples into synthetic variants
**So that** I can scale from 20 to 500+ samples while maintaining quality

### Acceptance Criteria
- [ ] Create `tools/generate_instruction_data.py`
- [ ] Support Azure OpenAI (existing env vars)
- [ ] Generate country-specific variants from templates
- [ ] Maintain answer structure and cautious language
- [ ] Include quality validation before saving
- [ ] Support resume/incremental generation (via seed)

### Technical Details
**Pipeline Steps:**
1. Load gold samples + question templates + country rules
2. For each template:
   - Generate question variant (paraphrase + country-specific values)
   - Generate structured answer using country rules
   - Apply quality checks (schema, tone, disclaimers)
   - Save to `data/instruction/synthetic.jsonl`
3. Track generation metadata (template_id, country, seed, model)

**LLM Prompting Strategy:**
```python
# Step 1: Generate question variant
question_prompt = f"""
Based on this template: "{template}"
Generate a natural German accounting question for {country}.
Keep it realistic and professional.
"""

# Step 2: Generate structured answer
answer_prompt = f"""
You are a professional accounting assistant for {country}.

Question: {question}

Country-specific rules:
{country_rules_context}

Generate a response with this structure:
1. Kurzantwort
2. Begründung (with legal reference)
3. Voraussetzungen/Prüfungen
4. Risiken/Ausnahmen
5. Hinweis/Disclaimer

Use cautious language: "grundsätzlich", "in der Regel", "abhängig vom Einzelfall"
Always include: "Diese Darstellung ersetzt keine steuerliche Beratung."
"""
```

**Quality Validation:**
- Schema compliance
- Disclaimer presence
- Legal reference format
- Cautious language phrases
- No absolute statements
- Professional tone

**Estimated Effort:** 5-7 days
**Dependencies:** Story 1, Story 2
**Output:** `tools/generate_instruction_data.py`, `data/instruction/synthetic.jsonl`

---

## Story 4: Country-Specific Variant Generator

**As a** data engineer
**I want** to automatically generate AT/DE/CH variants of each instruction sample
**So that** the model learns to respond correctly for each country's legal framework

### Acceptance Criteria
- [ ] Create `tools/expand_country_variants.py`
- [ ] For each gold sample, generate country-specific variants
- [ ] Replace legal references with country-appropriate paragraphs
- [ ] Adjust thresholds and values per country rules
- [ ] Tag each sample with `country` metadata field
- [ ] Validate factual correctness per country

### Technical Details
**Approach:**
1. Analyze gold sample topic
2. Check if topic has country-specific rules
3. If yes, generate 3 variants (AT, DE, CH)
4. If no, keep as DACH-generic with note

**Example Transformation:**
```python
# Gold sample (DE-focused)
{
  "id": "gold_003",
  "topic": "gwg",
  "messages": [
    {"role": "user", "content": "Was gilt als GWG?"},
    {"role": "assistant", "content": "...bis 800 Euro netto...§ 6 Abs. 2 EStG..."}
  ]
}

# Generated variants
{
  "id": "gold_003_AT",
  "country": "AT",
  "derived_from": "gold_003",
  "messages": [
    {"role": "user", "content": "Was gilt in Österreich als GWG?"},
    {"role": "assistant", "content": "...bis 800 Euro netto...§ 13 EStG (AT)..."}
  ]
}

{
  "id": "gold_003_CH",
  "country": "CH",
  "derived_from": "gold_003",
  "messages": [
    {"role": "user", "content": "Was gilt in der Schweiz als GWG?"},
    {"role": "assistant", "content": "...bis CHF 1'000...Art. 29 DBG..."}
  ]
}
```

**Estimated Effort:** 3-4 days
**Dependencies:** Story 1, existing gold samples
**Output:** `tools/expand_country_variants.py`, expanded `data/instruction/gold_country_variants.jsonl`

---

## Story 5: Quality Validation & Review Pipeline

**As a** data engineer
**I want** automated quality checks and human review workflows
**So that** only high-quality, factually correct samples enter the training set

### Acceptance Criteria
- [ ] Create `tools/validate_instruction_data.py`
- [ ] Automated checks: schema, disclaimers, legal refs, tone
- [ ] Flagging system for human review
- [ ] Review interface (simple CLI or notebook)
- [ ] Approval tracking in metadata
- [ ] Rejection reason logging

### Technical Details
**Validation Checks:**
```python
def validate_instruction_sample(sample):
    checks = {
        "schema_valid": validate_schema(sample),
        "has_disclaimer": check_disclaimer(sample),
        "has_legal_ref": check_legal_reference(sample) if required,
        "cautious_language": check_cautious_phrases(sample),
        "no_absolutes": check_forbidden_phrases(sample),
        "structure_present": check_answer_structure(sample),
        "country_consistency": check_country_refs(sample)
    }
    return checks

def flag_for_review(sample, failed_checks):
    """Flag samples that fail automated checks for human review"""
    pass
```

**Review Workflow:**
```bash
# Generate samples
python tools/generate_instruction_data.py --templates all --count 100

# Validate
python tools/validate_instruction_data.py --input data/instruction/synthetic.jsonl

# Review flagged samples
python tools/review_samples.py --flagged-only

# Approve/reject
# Updates meta.reviewed = true/false + meta.review_notes
```

**Metrics to Track:**
- Total samples generated
- Auto-validation pass rate
- Human review approval rate
- Common rejection reasons
- Coverage by topic/country/difficulty

**Estimated Effort:** 3-4 days
**Dependencies:** Story 3
**Output:** `tools/validate_instruction_data.py`, `tools/review_samples.py`

---

## Story 6: Paraphrasing & Diversity Expansion

**As a** data engineer
**I want** to generate diverse phrasings of the same question/answer
**So that** the model generalizes better and doesn't overfit to specific wordings

### Acceptance Criteria
- [ ] Create `tools/paraphrase_instruction_data.py`
- [ ] Generate 3-5 paraphrases per approved sample
- [ ] Maintain semantic equivalence
- [ ] Preserve professional tone and structure
- [ ] Track paraphrase lineage (derived_from)
- [ ] Human spot-check validation (10% sample)

### Technical Details
**Paraphrasing Strategy:**
```python
paraphrase_prompt = f"""
Paraphrase this accounting question in {n} different ways.
Keep the meaning identical.
Maintain professional German accounting terminology.
Vary sentence structure and wording.

Original: "{question}"

Generate {n} paraphrases:
"""

# For answers, preserve structure but vary wording
answer_paraphrase_prompt = f"""
Rewrite this accounting response with different wording.
PRESERVE:
- Structure (Kurzantwort → Begründung → ...)
- Legal references (exact § citations)
- Cautious language phrases
- Disclaimer

VARY:
- Sentence structure
- Connecting words
- Explanations (different examples/wording)

Original: "{answer}"
"""
```

**Quality Control:**
- Semantic similarity check (embedding cosine similarity > 0.9)
- Structure preservation check
- Legal reference preservation
- Tone consistency check

**Estimated Effort:** 2-3 days
**Dependencies:** Story 3, Story 5
**Output:** `tools/paraphrase_instruction_data.py`

---

## Story 7: Dataset Mixing & Splitting

**As a** data engineer
**I want** to create train/validation/test splits with proper stratification
**So that** training data is balanced and evaluation is meaningful

### Acceptance Criteria
- [ ] Create `tools/split_instruction_data.py`
- [ ] Support stratified splitting by topic/country/difficulty
- [ ] Ensure gold samples are overrepresented in training
- [ ] Create separate validation set for evaluation
- [ ] Generate dataset statistics and coverage report
- [ ] Support configurable split ratios

### Technical Details
**Splitting Strategy:**
```python
# Dataset composition
datasets = {
    "gold": load_jsonl("data/instruction/gold.jsonl"),  # 20 samples
    "gold_variants": load_jsonl("data/instruction/gold_country_variants.jsonl"),  # ~60 samples
    "synthetic": load_jsonl("data/instruction/synthetic.jsonl"),  # 500-2000 samples
    "paraphrased": load_jsonl("data/instruction/paraphrased.jsonl")  # 1000+ samples
}

# Stratification
split_config = {
    "train": 0.85,
    "val": 0.10,
    "test": 0.05,
    "stratify_by": ["topic", "difficulty", "country"],
    "gold_sampling_weight": 5.0  # Oversample gold in training
}

# Output
output = {
    "train": "data/instruction/train.jsonl",
    "val": "data/instruction/val.jsonl",
    "test": "data/instruction/test.jsonl"
}
```

**Coverage Report:**
```
Dataset Statistics
==================
Total Samples: 1,847

By Split:
- Train: 1,570 (85.0%)
- Val:     185 (10.0%)
- Test:     92 (5.0%)

By Topic:
- ust:            587 (31.8%)
- abschreibung:   324 (17.5%)
- belege:         289 (15.6%)
...

By Country:
- AT:     615 (33.3%)
- DE:     617 (33.4%)
- CH:     615 (33.3%)

By Difficulty:
- basic:        738 (40.0%)
- intermediate: 831 (45.0%)
- advanced:     278 (15.0%)
```

**Estimated Effort:** 2 days
**Dependencies:** All previous stories
**Output:** `tools/split_instruction_data.py`, train/val/test splits

---

## Story 8: Documentation & Usage Guide

**As a** future team member
**I want** clear documentation on how to generate and validate instruction data
**So that** I can extend the dataset or reproduce the pipeline

### Acceptance Criteria
- [ ] Create `docs/instruction_data_generation.md`
- [ ] Document end-to-end pipeline
- [ ] Include usage examples for all scripts
- [ ] Provide configuration guide
- [ ] Document quality standards and validation
- [ ] Include troubleshooting section

### Technical Details
**Documentation Sections:**
1. **Overview & Architecture**
2. **Prerequisites** (Azure OpenAI, environment setup)
3. **Step-by-Step Guide**
   - Generate country rules
   - Create question templates
   - Run synthetic expansion
   - Generate country variants
   - Validate & review
   - Paraphrase for diversity
   - Split datasets
4. **Configuration Reference**
5. **Quality Standards**
6. **Troubleshooting**

**Example Usage:**
```bash
# 1. Generate synthetic samples from templates
python tools/generate_instruction_data.py \
  --templates data/instruction/question_templates.json \
  --country-rules data/instruction/country_rules.json \
  --output data/instruction/synthetic.jsonl \
  --count 500 \
  --model gpt-4

# 2. Generate country-specific variants
python tools/expand_country_variants.py \
  --input data/instruction/gold.jsonl \
  --country-rules data/instruction/country_rules.json \
  --output data/instruction/gold_country_variants.jsonl

# 3. Validate
python tools/validate_instruction_data.py \
  --input data/instruction/synthetic.jsonl \
  --report validation_report.json

# 4. Review flagged samples
python tools/review_samples.py \
  --input data/instruction/synthetic.jsonl \
  --flagged-only

# 5. Paraphrase approved samples
python tools/paraphrase_instruction_data.py \
  --input data/instruction/synthetic.jsonl \
  --output data/instruction/paraphrased.jsonl \
  --variants 3

# 6. Create train/val/test splits
python tools/split_instruction_data.py \
  --inputs gold.jsonl,gold_country_variants.jsonl,synthetic.jsonl,paraphrased.jsonl \
  --output-dir data/instruction/splits/ \
  --stratify topic,country,difficulty
```

**Estimated Effort:** 1-2 days
**Dependencies:** All previous stories
**Output:** `docs/instruction_data_generation.md`

---

## Epic Summary

### Timeline Estimate
Total: **~25-35 days** (assuming 1 engineer, serial execution)

Parallel execution with 2-3 engineers: **~15-20 days**

### Deliverables

**Data Assets:**
- `data/instruction/country_rules.json` - Country-specific knowledge base
- `data/instruction/question_templates.json` - Template library
- `data/instruction/gold_country_variants.jsonl` - Country-specific gold samples
- `data/instruction/synthetic.jsonl` - LLM-generated synthetic samples
- `data/instruction/paraphrased.jsonl` - Paraphrased variants
- `data/instruction/train.jsonl` - Training split
- `data/instruction/val.jsonl` - Validation split
- `data/instruction/test.jsonl` - Test split

**Tools/Scripts:**
- `tools/generate_instruction_data.py` - Main synthetic generation
- `tools/expand_country_variants.py` - Country variant generator
- `tools/validate_instruction_data.py` - Quality validation
- `tools/review_samples.py` - Human review interface
- `tools/paraphrase_instruction_data.py` - Diversity expansion
- `tools/split_instruction_data.py` - Dataset splitting

**Documentation:**
- `docs/instruction_data_generation.md` - Complete usage guide

### Success Metrics
- ✅ 500-2,000 high-quality instruction samples
- ✅ ≥95% human review approval rate
- ✅ Full AT/DE/CH coverage for country-specific topics
- ✅ Balanced topic/difficulty/country distribution
- ✅ Schema compliance: 100%
- ✅ Validation pass rate: ≥98%

### Risk Mitigation

**Risk 1: LLM hallucination of legal rules**
*Mitigation:* Inject country rules as hard context, validate legal references

**Risk 2: Country-specific rules accuracy**
*Mitigation:* Domain expert review of country_rules.json

**Risk 3: Low quality synthetic samples**
*Mitigation:* Multi-stage validation + human review loop

**Risk 4: Insufficient coverage**
*Mitigation:* Template library + coverage metrics tracking

---

## Next Steps

1. **Review & Prioritize** this epic with stakeholders
2. **Story 1 (Country Rules)** - Start with domain expert collaboration
3. **Story 2 (Templates)** - Can be done in parallel with Story 1
4. **Stories 3-6** - Sequential execution recommended
5. **Story 7-8** - Final polish and documentation

## Notes

- All scripts should use existing Azure OpenAI configuration
- Maintain audit trail (generation metadata) for all synthetic samples
- Consider cost optimization (use GPT-4 for answers, GPT-3.5 for paraphrasing)
- Plan for iterative refinement based on initial results
