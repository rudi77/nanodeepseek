# Self-Maintaining Documentation
This file should be updated automatically when project-specific patterns, conventions, or important information are discovered during work sessions. Add relevant details here to help future interactions understand the codebase better. Anything that is very general knowledge about this project and that should be remembered always should be added here.

## Project notes (discovered)

- **Synthetic training data direction**: `docs/data_generation.md` describes generating bookkeeping SFT + DPO by keeping **accounting correctness rule-based** and using an LLM only to **paraphrase the task text** (JSON-only).
- **Ground-truth text corpora**: curated domain markdown exists under `data/processed/`:
  - **Laws**: `data/processed/1_fachtexte_regelwerke/*.md` (UGB, UStG, BAO, GoBD, etc.)
  - **Exams + solutions**: `data/processed/5_pruefungen/*.md`
- **Synthetic dataset tooling**: generators/validators/splitters live under `tools/` (EB generator, exam/law ingesters, split and mix utilities).


