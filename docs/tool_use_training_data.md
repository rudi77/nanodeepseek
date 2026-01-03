# Tool-Use Training Data für Buchhaltungs-LLM

## Überblick

Dieses Dokument beschreibt die Generierung von synthetischen Trainingsdaten für **Tool-Verwendung** (Tool Use / Function Calling) im Kontext eines Buchhaltungsassistenten.

## Motivation

Ein Buchhaltungsassistent muss nicht nur Buchungssätze erstellen, sondern auch:

- **Berechnungen** durchführen (USt, Skonto, Rundung)
- **Konten nachschlagen** (SKR03, SKR04, EKR)
- **Datumsoperationen** berechnen (Zahlungsfristen, Quartalsende)
- **USt-Sätze** ermitteln (nach Land und Jahr)
- **Buchungen validieren** (Soll/Haben Balance)

Diese Fähigkeiten erfordern **Tool-Calls** - das Modell muss lernen, wann und wie es externe Tools verwendet.

## Architektur

### 1. Tool-Definitionen (`tool_definitions.json`)

Definiert 5 Kerntools:

1. **calculator** - Mathematische Berechnungen (USt, Prozent, Rundung)
2. **account_lookup** - Kontensuche in SKR03/SKR04/EKR
3. **date_calculator** - Datumsoperationen (Fristen, Perioden-Ende)
4. **vat_lookup** - USt-Satz Abfrage (AT, DE, CH)
5. **balance_checker** - Soll/Haben Validierung

Jedes Tool hat ein JSON-Schema mit Parametern und Beschreibung.

### 2. Template-Library (`case_library_tool_use_100.json`)

25 Templates für verschiedene Tool-Use-Szenarien:

- **Calculation** (TU-001 bis TU-005): USt-Berechnungen, Rundung, Skonto
- **Account Lookup** (TU-006 bis TU-010): Kontensuche nach Name
- **Date Operations** (TU-011 bis TU-015): Zahlungsfristen, Perioden-Ende
- **VAT Lookup** (TU-016 bis TU-019): USt-Sätze nach Land/Datum
- **Validation** (TU-020): Balance-Checks
- **Combined** (TU-021 bis TU-025): Multi-Tool Szenarien

### 3. Generator (`tool_use_generator.py`)

**Regelbasierte Tool-Implementierung:**

Alle Tools sind **regelbasiert** implementiert - keine LLM-Calls für die Tool-Ausführung!

```python
execute_calculator("vat_gross", "1200 * 0.2", 2)
# → {"result": 1440.0, "calculation": "1200 * (1 + 0.2) = 1440.00"}

execute_account_lookup("Bank", "SKR03")
# → {"accounts": [{"name": "Bank", "number": "1200", "chart": "SKR03"}]}

execute_date_calculator("2024-01-15", "add_days", 30)
# → {"result": "2024-02-14", "base_date": "2024-01-15"}
```

**Paraphrasierung (Optional):**

- LLM (Azure GPT) wird **nur** für Aufgaben-Paraphrasierung verwendet
- Fallback auf Template-Text bei fehlenden Credentials
- `--no-paraphrase` Flag für vollständig LLM-freie Generierung

**Conversation Flow:**

```
User: "Berechne den Bruttobetrag für 1200 EUR netto bei 20% USt."
  ↓
Assistant: [calls calculator tool]
  tool_calls: [{
    "tool": "calculator",
    "arguments": {
      "operation": "vat_gross",
      "expression": "1200 * 0.2"
    }
  }]
  ↓
Tool: {"result": 1440.0, "calculation": "1200 * (1 + 0.2) = 1440.00"}
  ↓
Assistant: "Der Bruttobetrag beträgt 1440.00 EUR."
```

## Datenformate

### SFT (Supervised Fine-Tuning)

Multi-Turn Conversation mit Tool-Calls:

```json
{
  "schema_version": "sft.chat.v1",
  "messages": [
    {"role": "system", "content": "Du bist ein Buchhaltungsassistent..."},
    {"role": "user", "content": "Berechne den Bruttobetrag für 1200 EUR netto bei 20% USt."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "schema_version": "tool_call.v1",
        "tool": "calculator",
        "arguments": {"operation": "vat_gross", "expression": "1200 * 0.2"}
      }]
    },
    {
      "role": "tool",
      "content": "{\"schema_version\": \"tool_response.v1\", \"tool\": \"calculator\", \"success\": true, \"result\": {...}}"
    },
    {"role": "assistant", "content": "Der Bruttobetrag beträgt 1440.00 EUR."}
  ],
  "meta": {
    "source": "synthetic_tool_use",
    "template_id": "TU-001",
    "has_tool_calls": true
  }
}
```

### DPO (Direct Preference Optimization)

Preference Pairs mit korrekten vs. fehlerhaften Tool-Calls:

```json
{
  "schema_version": "dpo.v1",
  "prompt": "Berechne den Bruttobetrag für 1200 EUR netto bei 20% USt.",
  "chosen": "[...messages with correct tool call...]",
  "rejected": "[...messages with WRONG tool call...]",
  "meta": {
    "error_class": "WRONG_PARAM",
    "has_tool_calls": true
  }
}
```

**Error Classes für DPO:**

- `WRONG_TOOL` - Falsches Tool gewählt
- `WRONG_PARAM` - Korrupte Parameter (z.B. "INVALID" statt Expression)
- `MISSING_PARAM` - Fehlende Required-Parameter

## Verwendung

### Generierung ohne LLM (nur Regeln)

```bash
python tools/generate_tool_use_dataset.py \
  --sft-count 1000 \
  --dpo-count 1000 \
  --no-paraphrase
```

**Output:**
- `train_sft_tool_use_1000.jsonl` (1000 Zeilen)
- `train_dpo_tool_use_1000.jsonl` (1000 Zeilen)

### Generierung mit Azure GPT Paraphrasierung

```bash
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="gpt-4"

python tools/generate_tool_use_dataset.py \
  --sft-count 1000 \
  --dpo-count 1000
```

### Optionen

```bash
--templates PATH              # Template-Library (default: case_library_tool_use_100.json)
--out-sft PATH                # SFT Output (default: train_sft_tool_use_1000.jsonl)
--out-dpo PATH                # DPO Output (default: train_dpo_tool_use_1000.jsonl)
--seed INT                    # Random Seed (default: 42)
--sft-count INT               # Anzahl SFT Samples (default: 1000)
--dpo-count INT               # Anzahl DPO Samples (default: 1000)
--no-paraphrase               # Keine LLM-Paraphrasierung
```

## Validierung

Validator für Tool-Use Daten:

```python
from validator import validate_sft_tool_use_row, validate_dpo_tool_use_row

ok, errors = validate_sft_tool_use_row(sft_row)
# Checks:
# - Schema version
# - Message structure
# - Tool call format
# - Tool response format

ok, errors = validate_dpo_tool_use_row(dpo_row)
# Checks:
# - Chosen/Rejected JSON validity
# - Message structure
```

## Qualitätsmetriken

**Erwartete Werte:**

- SFT Parse Rate: **100%** (regelbasiert, deterministisch)
- DPO Parse Rate: **100%** (regelbasiert, deterministisch)
- Tool Call Success Rate: **100%** (für SFT chosen)
- DPO Error Injection Rate: **100%** (für rejected)

**Vorteile:**

1. **Deterministische Korrektheit** - Tool-Ausführung ist regelbasiert
2. **Keine Halluzinationen** - Keine numerischen Fehler
3. **Skalierbar** - Kann ohne LLM-Kosten generiert werden
4. **Validierbar** - Klare Fehlerklassen für DPO

## Erweiterung

### Neue Tools hinzufügen

1. Tool in `tool_definitions.json` definieren
2. Implementierung in `tool_use_generator.py::execute_tool()`
3. Templates in `case_library_tool_use_100.json` erstellen

### Multi-Tool Templates

Für komplexe Szenarien mit mehreren sequenziellen Tool-Calls:

```json
{
  "template_id": "TU-XXX",
  "tool_sequence": ["vat_lookup", "calculator", "account_lookup"],
  ...
}
```

## Integration in Training-Pipeline

Mischen Sie Tool-Use Daten mit anderen Datasets:

```bash
python tools/mix_datasets.py --config tools/mix_config_with_tools.json
```

Beispiel `mix_config_with_tools.json`:

```json
{
  "datasets": {
    "eb": {"path": "train_sft_eb_1000.jsonl", "weight": 0.4},
    "exams": {"path": "train_sft_exams.jsonl", "weight": 0.3},
    "tools": {"path": "train_sft_tool_use_1000.jsonl", "weight": 0.3}
  },
  "output": "train_sft_all_with_tools.jsonl"
}
```

## Limitierungen

1. **Multi-Tool Templates** - Aktuell nur begrenzt implementiert
2. **Account Mapping** - Nur vereinfachtes SKR03-Mapping
3. **VAT Rates** - Statische Sätze, keine historischen Änderungen
4. **Date Calculator** - Vereinfachte Monatsberechnung (30 Tage)

## Nächste Schritte

- [ ] Multi-Tool Template Implementierung vervollständigen
- [ ] Erweiterte Kontenrahmen (SKR04, vollständiges EKR)
- [ ] Historische USt-Sätze (z.B. DE 2020 Corona-Absenkung)
- [ ] Komplexere Berechnungen (Abschreibungen, Rückstellungen)
- [ ] Tool-Error-Handling (z.B. "Konto nicht gefunden" Szenarien)

## Beispiel-Output

**SFT Sample:**

```json
{
  "messages": [
    {"role": "system", "content": "Du bist ein Buchhaltungsassistent..."},
    {"role": "user", "content": "Wie hoch ist der Normalsteuersatz in Österreich am 2024-01-15?"},
    {
      "role": "assistant",
      "tool_calls": [{
        "tool": "vat_lookup",
        "arguments": {"country": "AT", "date": "2024-01-15", "category": "normal"}
      }]
    },
    {
      "role": "tool",
      "content": "{\"country\": \"AT\", \"rate\": 0.20, \"rate_percent\": 20}"
    },
    {"role": "assistant", "content": "Der USt-Satz beträgt 20% (0.2)."}
  ]
}
```

**DPO Sample (Rejected):**

```json
{
  "rejected": [
    {"role": "user", "content": "Wie hoch ist der Normalsteuersatz..."},
    {
      "role": "assistant",
      "tool_calls": [{
        "tool": "wrong_tool",  // ❌ WRONG_TOOL
        "arguments": {"country": "AT"}
      }]
    },
    ...
  ]
}
```

## Lizenz

Siehe Hauptprojekt-Lizenz.
