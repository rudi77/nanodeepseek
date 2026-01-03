# Synthetic Training Data Schemas (v1)

This document freezes the dataset envelopes and the canonical BookEntry schema.

## SFT Chat JSONL Envelope (sft.chat.v1)

One JSON object per line:

- schema_version: "sft.chat.v1"
- messages: list of {role, content} objects
  - messages[0] system prompt
  - messages[1] user task/instruction
  - messages[2] assistant JSON string containing BookEntry v1
- meta: provenance and generation metadata

Example:

```
{
  "schema_version": "sft.chat.v1",
  "messages": [
    {"role": "system", "content": "Du bist Buchhaltungsassistent (AT). Antworte ausschliesslich mit JSON."},
    {"role": "user", "content": "Erstelle den Buchungssatz: Bankguthaben laut Kontoauszug. Datum: 2025-01-03. Branche: all. Betrag: 1200.00."},
    {"role": "assistant", "content": "{\"schema_version\":\"bookentry.v1\",\"datum\":\"2025-01-03\",\"industry\":\"all\",\"template_id\":\"EB-001\",\"text\":\"Erstelle den Buchungssatz: ...\",\"lines\":[{\"account_label\":\"Bank\",\"side\":\"Soll\",\"amount\":1200.0},{\"account_label\":\"Eroeffnungsbilanzkonto\",\"side\":\"Haben\",\"amount\":1200.0}]}"}
  ],
  "meta": {
    "source": "synthetic_template",
    "template_id": "EB-001",
    "industry": "all",
    "seed": 42,
    "amount_display": 1200.0,
    "amount_bucket": 12
  }
}
```

## DPO JSONL Envelope (dpo.v1)

One JSON object per line:

- schema_version: "dpo.v1"
- prompt: instruction text
- chosen: JSON string for correct BookEntry v1
- rejected: JSON string for incorrect BookEntry v1
- meta: provenance and error_class

Example:

```
{
  "schema_version": "dpo.v1",
  "prompt": "Erstelle den Buchungssatz: ...",
  "chosen": "{\"schema_version\":\"bookentry.v1\",...}",
  "rejected": "{\"schema_version\":\"bookentry.v1\",...}",
  "meta": {
    "source": "synthetic_template",
    "template_id": "EB-001",
    "error_class": "SWAP_SIDE",
    "seed": 42
  }
}
```

## BookEntry Schema (bookentry.v1)

Canonical output for Er?ffnungsbuchungen (EB):

- schema_version: "bookentry.v1"
- datum: YYYY-MM-DD
- industry: string
- template_id: string
- text: string (instruction or short description)
- lines: list of booking lines
  - account_label: string
  - side: "Soll" or "Haben"
  - amount: number with 2 decimals
  - ekr_code: optional string

Example:

```
{
  "schema_version": "bookentry.v1",
  "datum": "2025-01-03",
  "industry": "all",
  "template_id": "EB-001",
  "text": "Erstelle den Buchungssatz: Bankguthaben laut Kontoauszug.",
  "lines": [
    {"account_label": "Bank", "side": "Soll", "amount": 1200.0},
    {"account_label": "Eroeffnungsbilanzkonto", "side": "Haben", "amount": 1200.0}
  ]
}
```

## Versioning Rules

- Any breaking schema change requires bumping schema_version.
- Only JSON objects are allowed in assistant outputs (no Markdown).
- Validation must be deterministic for SFT and DPO rows.
