# Claude Code Guidelines for NanoDeepSeek

This file provides guidance for working with the NanoDeepSeek project using Claude Code. It should be updated as project conventions and patterns evolve.

## Project Overview

**NanoDeepSeek** is a synthetic training data generation pipeline for German/Austrian (DACH) accounting models. The project focuses on creating high-quality training data for:
- **SFT** (Supervised Fine-Tuning) in chat JSONL format
- **DPO** (Direct Preference Optimization) with prompt/chosen/rejected pairs

### Core Principle
- **Correctness is rule-based**: Accounting math and booking logic are computed deterministically
- **LLM for paraphrasing only**: Language models are used exclusively for natural-language task text, NOT for bookkeeping solutions

## Project Structure

### Key Directories
- `data/processed/`: Curated domain markdown (laws, exams, regulations)
  - `1_fachtexte_regelwerke/`: Austrian/German laws (UGB, UStG, BAO, GoBD, etc.)
  - `5_pruefungen/`: Exam materials with solutions
- `data/instruction/`: Instruction-layer training data for assistant behavior
  - `gold.jsonl`: 20 manually curated, high-quality instruction samples
- `docs/`: Project documentation
  - `prd_synthetic_training_data.md`: Product requirements for booking/tool-use data
  - `prd_instruction_data.md`: Product requirements for instruction/assistant behavior data
  - `tool_use_training_data.md`: Tool-use training data specification
  - `data_generation.md`: Data generation methodology
  - `epics_synthetic_training_data.md`: Feature epics
  - `schemas_synthetic_training_data.md`: Schema definitions
- `tools/`: Generators, validators, splitters
  - EB (Eröffnungsbuchungen) generator
  - Exam/law ingesters
  - Split and mix utilities

### Key Files
- `case_library_eb_100.json`: 100 Eröffnungsbuchung templates (EB-001 to EB-100)
- `generate_eb_dataset.py`: Main generator script for EB datasets

## Development Conventions

### Code Style
- Use Python for data generation scripts
- Environment: Project uses `uv` for Python package management
- Required environment variables for Azure OpenAI:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT`
  - (optional) `AZURE_OPENAI_API_VERSION`

### Data Formats

The project uses two main types of training data:
1. **Booking/Tool-Use Data**: Transaction-level correctness (bookings, calculations)
2. **Instruction Data**: Assistant behavior (tone, structure, safety, interaction)

#### Instruction Dataset (assistant behavior JSONL)
- One JSON object per line
- Structure:
  ```json
  {
    "id": "gold_001",
    "type": "instruction",
    "source": "manual | script | law | mixed",
    "topic": "ust | abschreibung | belege | reisekosten | bewirtung | gwg | allgemein",
    "language": "de",
    "messages": [
      {"role": "user", "content": "Clear, realistic accounting question"},
      {"role": "assistant", "content": "Structured, cautious, factually correct response"}
    ],
    "meta": {
      "difficulty": "basic | intermediate | advanced",
      "contains_legal_reference": true,
      "reviewed": true
    }
  }
  ```

**Quality requirements for assistant responses:**
- **Structure**: Kurzantwort → Begründung → Voraussetzungen → Risiken → Hinweis
- **Cautious language**: "grundsätzlich", "in der Regel", "abhängig vom Einzelfall"
- **Professional tone**: Sachlich, ruhig, professionell, unterstützend
- **Legal references**: Cite specific paragraphs (e.g., "§ 15 Abs. 1 UStG")
- **Disclaimers**: "Diese Darstellung ersetzt keine steuerliche Beratung."

#### SFT Dataset (chat JSONL)
- One JSON object per line
- Structure:
  ```json
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "{JSON booking object}"}
    ],
    "meta": {
      "template_id": "EB-001",
      "industry": "...",
      "source": "synthetic_template"
    }
  }
  ```

#### DPO Dataset (JSONL)
- One JSON object per line
- Structure:
  ```json
  {
    "prompt": "instruction text",
    "chosen": "{correct JSON solution}",
    "rejected": "{plausible wrong JSON solution}",
    "meta": {
      "template_id": "EB-001",
      "error_class": "swap_sides",
      "industry": "...",
      "seed": 42
    }
  }
  ```

#### Canonical Booking Output Schema (v1)
```json
{
  "datum": "2024-01-15",
  "industry": "...",
  "template_id": "EB-001",
  "text": "...",
  "lines": [
    {
      "account_label": "...",
      "side": "Soll",
      "amount": 1234.56,
      "ekr_code": "..."
    },
    {
      "account_label": "...",
      "side": "Haben",
      "amount": 1234.56,
      "ekr_code": "..."
    }
  ]
}
```

### Validation Requirements
- **Parse rate**: ≥99% for all JSON outputs
- **Validation pass rate**: ≥98% for chosen samples
- **Balance requirement**: Soll == Haben (sum of debit == sum of credit)
- **VAT math**: Correct when applicable
- **Amounts**: Positive, rounded to 2 decimals
- **Schema**: All required keys present, enums valid

### Quality Standards
- **Deterministic correctness**: Same seed + config = identical output
- **Controlled errors**: Rejected samples contain exactly one error class where feasible
- **Provenance tracking**: Every sample includes source and identifiers for audit/replay
- **No PII**: Use synthetic company names only

## Git Workflow

### Branch Strategy
- Feature branches: Use `claude/` prefix with session ID
- Main development branch: Check git status for current branch
- Always develop on designated branch
- Never push to different branches without permission

### Commit Guidelines
- Clear, descriptive commit messages
- Commit when explicitly requested or when work is complete
- Use proper formatting for multi-line commit messages (HEREDOC)

## Common Tasks

### Generating Datasets
1. Ensure environment variables are set (Azure OpenAI)
2. Run generation script: `python generate_eb_dataset.py`
3. Validate outputs using validation rules
4. Check parse rates and quality metrics

### Adding New Templates
- Add to `case_library_eb_100.json`
- Include: template_id, category, sub_category, industry_focus, description, amount_model, booking, rules
- Ensure rule-based correctness

### Working with Laws/Exams
- Source files in `data/processed/`
- Extract structured pairs (question/solution)
- Normalize into canonical JSON where possible
- Apply paraphrasing and controlled errors for DPO

## Important Notes

### Data Sources Priority
1. **Template library**: Primary source for synthetic data (case_library_eb_100.json)
2. **Curated texts**: Laws and exams for grounding and realism
3. **Paraphrasing**: LLM-generated variations for instruction text only

### Scope
- **In scope**: Austrian/DACH accounting, SFT/DPO generation, validation
- **Out of scope**: End-user UI, full RAG systems, multi-country tax engines (initial focus: AT/DACH)

### Error Handling
- Log all validation failures with reasons
- Track rejected sample "wrongness" (≥95% should fail at least one validation rule)
- Maintain coverage metrics per template and industry

## Documentation Updates

When discovering new patterns, conventions, or important information:
1. Update `AGENTS.md` for general project knowledge
2. Update this file (`CLAUDE.md`) for Claude-specific workflows
3. Update relevant docs in `docs/` for detailed specifications

## References

- Booking/Tool-Use Data PRD: `docs/prd_synthetic_training_data.md`
- Instruction Data PRD: `docs/prd_instruction_data.md`
- Tool-Use Training Data: `docs/tool_use_training_data.md`
- Data generation methodology: `docs/data_generation.md`
- Schemas: `docs/schemas_synthetic_training_data.md`
- Epics: `docs/epics_synthetic_training_data.md`
