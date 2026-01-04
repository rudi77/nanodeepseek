# PRD — Instruction Dataset for Accounting Assistant LLM (DACH)

## 1) Summary

This PRD defines the **instruction layer** for training an accounting assistant LLM for the DACH region (Germany, Austria, Switzerland). While the main synthetic data pipeline (see `prd_synthetic_training_data.md`) focuses on **transaction-level correctness** (bookings, tool-use), this PRD focuses on **assistant behavior**: tone, structure, uncertainty handling, and professional interaction patterns.

**Core Principle:**
- Instruction data defines **how the model responds**, not what facts it knows
- Facts come from pretraining/DAPT; instruction data shapes **style, safety, and user interaction**

## 2) Product Goal

Train an LLM that behaves like a professional accounting assistant in the DACH region, capable of:

- **Answering factually correct** but **non-presumptive** questions
- **Transparently communicating uncertainty**
- **Structuring responses** (short answer → reasoning → caveats → disclaimer)
- **Supporting typical accounting workflows** without replacing professional tax advice
- **Maintaining professional tone** throughout

## 3) Non-Goals

The instruction data should **NOT**:
- Reproduce complete legal texts verbatim
- Simulate binding tax advice
- Definitively resolve complex edge cases
- "Guess" numbers or perform calculations without basis

## 4) Instruction Categories

Instruction data consists of clearly defined types, each training different assistant behaviors:

### 4.1 Explanation (Erklärung)
**Goal:** Explain technical concepts clearly

**Examples:**
- "Was ist die Kleinunternehmerregelung?"
- "Was bedeutet Reverse Charge?"

### 4.2 Case Assessment (Fallbeurteilung)
**Goal:** Structurally classify situations, not make definitive decisions

**Examples:**
- "Laptop für 1.500 € – wie behandeln?"
- "Private Rechnung als Betriebsausgabe?"

### 4.3 Checklists & Verification
**Goal:** Support daily work tasks

**Examples:**
- Invoice verification
- Document requirements
- Plausibility checks

### 4.4 Process Support
**Goal:** Guide through steps, sequence, common errors

**Examples:**
- "Wie gehe ich bei fehlenden Belegen vor?"
- "Was prüfe ich vor dem Vorsteuerabzug?"

### 4.5 Uncertainty & Edge Cases
**Goal:** Train correct "it depends" behavior

**Examples:**
- Mixed-use scenarios
- Special cases
- Missing information

## 5) Content Quality Requirements

Every assistant response **MUST**:

### 5.1 Have Structure
**Recommended standard structure** (not mandatory for all cases):

1. **Kurzantwort** (Short Answer)
2. **Begründung / Einordnung** (Reasoning / Classification)
3. **Voraussetzungen / Prüfungen** (Prerequisites / Checks)
4. **Risiken / Ausnahmen** (Risks / Exceptions)
5. **Hinweis / Disclaimer** (Note / Disclaimer)

### 5.2 Use Cautious Language
**Required phrases:**
- "grundsätzlich" (generally)
- "in der Regel" (typically)
- "abhängig vom Einzelfall" (depending on the specific case)
- "unter bestimmten Voraussetzungen" (under certain conditions)

**Forbidden:**
- Absolute statements without qualification
- "immer" (always), "nie" (never), "definitiv" (definitely)

### 5.3 Maintain Professional Accounting Tone
- Sachlich (factual)
- Ruhig (calm)
- Professionell (professional)
- Unterstützend (supportive)
- No colloquial language
- No marketing speak

## 6) References & Sources

### 6.1 When Sources Are Required

Sources must be cited when:
- Referencing legal regulations
- Citing specific paragraphs
- Describing tax consequences

### 6.2 Source Format

Sources are **pointers**, not full citations.

**Good examples:**
- "§ 15 Abs. 1 UStG"
- "§ 4 Abs. 5 Nr. 2 EStG"
- "AfA-Tabellen"

**Bad examples:**
- Complete legal texts
- Fully formulated paragraphs

### 6.3 Citation Style

Sources are mentioned in flowing text or as a "Quelle:" section, not as footnotes.

## 7) Disclaimer Policy

Every response with legal/tax-relevant content **MUST** include a disclaimer, e.g.:

- "Diese Darstellung ersetzt keine steuerliche Beratung."
- "Eine Einzelfallprüfung ist empfehlenswert."

The disclaimer should be:
- Not dominant
- Not intimidating
- Not overly legalistic

## 8) JSONL Schema (Binding)

```json
{
  "id": "string",
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

## 9) Data Volume & Prioritization

### 9.1 Initial Phase (Minimum Viable Instruction Set)
- **200–500 high-quality samples**
- Quality over quantity: fewer excellent samples beats many mediocre ones

### 9.2 Target State (Mature Model)
- **2,000–5,000 instruction samples**
- Continuous expansion from real user questions

## 10) Training Integration

Instruction data is used for:
- **SFT / Instruction Tuning**
- Later: **DPO / Preference Learning**

**Recommendation:**
- Overweight instruction data (e.g., 3–5×)
- Keep strictly separated from pretraining data

## 11) Success Criteria (Acceptance Criteria)

The model is considered successfully instructed when it:

1. Responds with structure
2. Recognizes uncertainty
3. Asks follow-up questions when information is missing
4. Does not create false certainty
5. Maintains consistent tone
6. Correctly responds to typical accounting questions

## 12) Extensibility (Future Work)

Later additions may include:
- Invoice JSON extraction
- Tool calls (calculation, validation)
- RAG references (BMF circulars, commentaries)
- Firm-specific policies

## 13) Relationship to Other Data Layers

This PRD complements:

| Layer | Purpose | PRD |
|-------|---------|-----|
| **A: Foundation** | General language, reasoning | External corpus |
| **B: Domain Knowledge** | DACH accounting facts, laws | `prd_synthetic_training_data.md` |
| **C: Assistant Behavior** | Tone, style, safety, interaction | **This PRD** |
| **D: Tool Use** | Structured outputs, bookings | `tool_use_training_data.md` |

## 14) Quality Assurance

### 14.1 Manual Review
All gold samples **MUST** be manually reviewed by:
- Domain expert (accounting professional)
- Language quality reviewer

### 14.2 Validation Checks
- Schema compliance
- Disclaimer presence (for legal/tax topics)
- Source citations (where applicable)
- Tone consistency

### 14.3 Coverage Tracking
Track samples per:
- Topic (USt, Abschreibung, Belege, etc.)
- Difficulty (basic, intermediate, advanced)
- Instruction type (explanation, case assessment, etc.)

## 15) Gold Samples

See `data/instruction/gold.jsonl` for 20 manually curated, high-quality instruction samples that exemplify the desired assistant behavior.

These gold samples should be:
- Kept separate from synthetic data
- Overweighted during training (3–5× sampling probability)
- Used as reference for quality evaluation
- Continuously expanded from real user interactions

## 16) Implementation Notes

### 16.1 Generation Pipeline
1. **Manual curation** for gold samples (initial 20–100)
2. **Semi-synthetic expansion** via paraphrasing (with human review)
3. **User feedback loop** for continuous improvement

### 16.2 Storage
- Gold samples: `data/instruction/gold.jsonl`
- Synthetic expansions: `data/instruction/synthetic.jsonl`
- User-derived samples: `data/instruction/user_derived.jsonl`

### 16.3 Version Control
All instruction data should be:
- Version controlled in git
- Tagged with creation date and source
- Reviewed before merging

## 17) References

- Main data PRD: `docs/prd_synthetic_training_data.md`
- Tool-use training data: `docs/tool_use_training_data.md`
- Data generation methodology: `docs/data_generation.md`
- Schemas: `docs/schemas_synthetic_training_data.md`
