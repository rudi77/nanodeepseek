# Instruction Data for Accounting Assistant LLM (DACH)

This directory contains instruction-layer training data for the NanoDeepSeek accounting assistant. These samples define **assistant behavior** (tone, structure, safety, interaction patterns), complementing the booking/tool-use data that focuses on transaction-level correctness.

## Overview

**Purpose:** Train an LLM to behave like a professional accounting assistant in the DACH region (Germany, Austria, Switzerland)

**Key Characteristics:**
- Factually correct but non-presumptive answers
- Transparent communication of uncertainty
- Structured responses (short answer → reasoning → caveats → disclaimer)
- Professional accounting tone throughout

## Files in This Directory

### `gold.jsonl`
20 manually curated, high-quality instruction samples that exemplify the desired assistant behavior.

**Coverage:**
- VAT (Umsatzsteuer)
- Depreciation (Abschreibung)
- Documents/Receipts (Belege)
- Travel Expenses (Reisekosten)
- Entertainment Expenses (Bewirtung)
- Low-Value Assets (GWG)
- General Accounting Topics

**Difficulty Levels:**
- Basic: 8 samples
- Intermediate: 9 samples
- Advanced: 3 samples

## JSONL Schema

Each line in `gold.jsonl` is a complete JSON object with the following structure:

```json
{
  "id": "gold_001",
  "type": "instruction",
  "source": "manual | script | law | mixed",
  "topic": "ust | abschreibung | belege | reisekosten | bewirtung | gwg | allgemein",
  "language": "de",
  "messages": [
    {
      "role": "user",
      "content": "Clear, realistic accounting question"
    },
    {
      "role": "assistant",
      "content": "Structured, cautious, factually correct response"
    }
  ],
  "meta": {
    "difficulty": "basic | intermediate | advanced",
    "contains_legal_reference": true,
    "reviewed": true
  }
}
```

## Quality Requirements

### Assistant Response Structure
Recommended structure (not mandatory for all cases):
1. **Kurzantwort** (Short Answer)
2. **Begründung / Einordnung** (Reasoning / Classification)
3. **Voraussetzungen / Prüfungen** (Prerequisites / Checks)
4. **Risiken / Ausnahmen** (Risks / Exceptions)
5. **Hinweis / Disclaimer** (Note / Disclaimer)

### Cautious Language
**Required phrases:**
- "grundsätzlich" (generally)
- "in der Regel" (typically)
- "abhängig vom Einzelfall" (depending on the specific case)
- "unter bestimmten Voraussetzungen" (under certain conditions)

**Forbidden:**
- Absolute statements without qualification
- "immer" (always), "nie" (never), "definitiv" (definitely)

### Professional Tone
- Sachlich (factual)
- Ruhig (calm)
- Professionell (professional)
- Unterstützend (supportive)
- No colloquial language
- No marketing speak

### Legal References
When citing laws, use specific paragraph references:
- Good: "§ 15 Abs. 1 UStG"
- Bad: Full legal text reproduction

### Disclaimers
Every response with legal/tax-relevant content must include a disclaimer:
- "Diese Darstellung ersetzt keine steuerliche Beratung."
- "Eine Einzelfallprüfung ist empfehlenswert."

## Usage in Training

### Recommended Approach
1. **Keep gold samples separate** from synthetic data
2. **Overweight during training** (3–5× sampling probability)
3. **Use as quality reference** for evaluation
4. **Continuously expand** from real user interactions

### Training Integration
- **SFT (Supervised Fine-Tuning):** Primary use case
- **DPO (Direct Preference Optimization):** Future extension

### Sampling Strategy
```python
# Example: Overweight gold samples during training
dataset = {
    "gold": load_jsonl("data/instruction/gold.jsonl"),
    "synthetic": load_jsonl("data/instruction/synthetic.jsonl")
}

# Sample gold samples 5x more frequently
training_data = sample_with_weights(
    datasets=dataset,
    weights={"gold": 5.0, "synthetic": 1.0}
)
```

## Coverage Breakdown

### By Topic
| Topic | Count | Examples |
|-------|-------|----------|
| USt (VAT) | 7 | Vorsteuerabzug, Kleinunternehmer, Reverse-Charge |
| Abschreibung | 3 | Laptop, Kfz-Privatanteil, Rückstellungen |
| Belege | 3 | Pflichtangaben, fehlerhafte Rechnung, fehlender Beleg |
| GWG | 2 | Definition, Sammelposten |
| Reisekosten | 2 | Inland, Ausland |
| Bewirtung | 1 | Steuerliche Behandlung |
| Allgemein | 2 | Periodenabgrenzung, "kommt darauf an"-Fall |

### By Difficulty
| Difficulty | Count | Percentage |
|------------|-------|------------|
| Basic | 8 | 40% |
| Intermediate | 9 | 45% |
| Advanced | 3 | 15% |

## Validation & Quality Assurance

### Manual Review Checklist
- [ ] Schema compliance
- [ ] Disclaimer present (for legal/tax topics)
- [ ] Legal references cited (where applicable)
- [ ] Cautious language used
- [ ] Professional tone maintained
- [ ] Factual correctness verified
- [ ] Structure appropriate for question type

### Automated Checks
```bash
# Validate JSONL format
python -m json.tool gold.jsonl > /dev/null && echo "Valid JSONL"

# Count samples
wc -l gold.jsonl

# Check for required fields
jq -r '.messages[1].content' gold.jsonl | grep -c "Hinweis"
```

## Extension Guidelines

### Adding New Samples
When adding new instruction samples:

1. **Follow the schema** strictly
2. **Maintain quality bar** (use gold samples as reference)
3. **Get domain expert review** for factual correctness
4. **Track coverage** by topic and difficulty
5. **Avoid duplication** of similar questions
6. **Document source** (manual, derived from user query, etc.)

### Future Expansions
- `synthetic.jsonl`: Semi-synthetic samples via paraphrasing
- `user_derived.jsonl`: Samples from real user interactions
- `edge_cases.jsonl`: Specialized samples for uncertainty handling

## References

- **PRD:** `docs/prd_instruction_data.md` - Full product requirements
- **Project Guidelines:** `CLAUDE.md` - Development conventions
- **Related Data:** `docs/prd_synthetic_training_data.md` - Booking/tool-use data

## License & Usage

This dataset is part of the NanoDeepSeek project for training DACH accounting assistant models. All samples are manually reviewed for quality and factual correctness.

**Note:** These samples represent professional accounting guidance but do not constitute binding tax advice. Always consult with qualified tax professionals for specific cases.
