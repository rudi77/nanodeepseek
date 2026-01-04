"""Expand gold instruction samples to AT/DE/CH country variants.

This script takes gold instruction samples (which are typically DE-focused) and
creates country-specific variants for Austria (AT), Germany (DE), and Switzerland (CH)
by adapting legal references, amounts, and country-specific language.

Usage:
    python tools/expand_country_variants.py \
        --input data/instruction/gold.jsonl \
        --country-rules data/instruction/country_rules.json \
        --output data/instruction/gold_expanded.jsonl \
        --countries AT,DE,CH

Required environment variables:
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import AzureOpenAI
except ImportError:
    print("Error: openai package not installed. Run: uv pip install openai")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


COUNTRY_NAMES = {
    "AT": "Österreich",
    "DE": "Deutschland",
    "CH": "der Schweiz"
}

# Country-specific legal frameworks
COUNTRY_LEGAL_SYSTEMS = {
    "AT": {
        "income_tax_act": "EStG",
        "vat_act": "UStG",
        "commercial_code": "UGB",
        "fiscal_code": "BAO"
    },
    "DE": {
        "income_tax_act": "EStG",
        "vat_act": "UStG",
        "commercial_code": "HGB",
        "fiscal_code": "AO"
    },
    "CH": {
        "income_tax_act": "DBG",
        "vat_act": "MWSTG",
        "commercial_code": "OR",
        "fiscal_code": "StHG"
    }
}


class CountryVariantExpander:
    """Expands instruction samples to country-specific variants."""

    def __init__(
        self,
        client: AzureOpenAI,
        deployment: str,
        country_rules: List[Dict[str, Any]],
        temperature: float = 0.3,
    ):
        self.client = client
        self.deployment = deployment
        self.country_rules = {r["rule_id"]: r for r in country_rules}
        self.temperature = temperature

    def expand_sample(
        self,
        sample: Dict[str, Any],
        target_country: str
    ) -> Optional[Dict[str, Any]]:
        """Expand a single sample to a target country variant."""

        # If sample is already country-specific and matches target, return as-is
        source_country = sample.get("country", "DE")
        if source_country == target_country:
            return sample

        # Extract original question and answer
        user_msg = None
        assistant_msg = None
        for msg in sample.get("messages", []):
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]

        if not user_msg or not assistant_msg:
            logger.warning(f"Sample {sample.get('id')} missing messages")
            return None

        # Detect topic to find relevant rules
        topic = sample.get("topic", "allgemein")
        relevant_rules = self._find_relevant_rules(topic, assistant_msg)

        # Build country-specific context
        context = self._build_country_context(relevant_rules, target_country)

        # Generate adapted answer
        adapted_answer = self._adapt_answer(
            user_msg,
            assistant_msg,
            target_country,
            context
        )

        if not adapted_answer:
            logger.warning(f"Failed to adapt sample {sample.get('id')} to {target_country}")
            return None

        # Adapt question if it contains country references
        adapted_question = self._adapt_question(user_msg, target_country)

        # Build new sample
        new_sample = {
            **sample,
            "id": f"{sample['id']}_{target_country}",
            "country": target_country,
            "language": "de",
            "messages": [
                {"role": "user", "content": adapted_question},
                {"role": "assistant", "content": adapted_answer}
            ],
            "generation_metadata": {
                "model": self.deployment,
                "temperature": self.temperature,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "source_sample_id": sample.get("id"),
                "source_country": source_country,
                "expansion_method": "country_variant"
            }
        }

        # Update source to indicate it's an expansion
        if new_sample.get("source") == "manual":
            new_sample["source"] = "manual_expanded"

        return new_sample

    def _find_relevant_rules(self, topic: str, content: str) -> List[str]:
        """Find relevant country rules based on topic and content."""
        relevant = []

        for rule_id, rule in self.country_rules.items():
            if rule.get("topic") == topic:
                relevant.append(rule_id)
            # Also check if content mentions this rule
            elif rule.get("description", "").lower() in content.lower():
                relevant.append(rule_id)

        return relevant

    def _build_country_context(self, rule_ids: List[str], country: str) -> str:
        """Build country-specific context from rules."""
        context_parts = []

        for rule_id in rule_ids:
            rule = self.country_rules.get(rule_id)
            if not rule:
                continue

            country_data = rule["countries"].get(country)
            if not country_data:
                continue

            context_parts.append(f"**{rule['description']}** ({country}):")
            for key, value in country_data.items():
                if isinstance(value, list):
                    context_parts.append(f"  - {key}:")
                    for item in value:
                        context_parts.append(f"    • {item}")
                else:
                    context_parts.append(f"  - {key}: {value}")
            context_parts.append("")

        return "\n".join(context_parts)

    def _adapt_question(self, question: str, target_country: str) -> str:
        """Adapt question to target country (simple replacement)."""
        # Replace country names
        for country_code, country_name in COUNTRY_NAMES.items():
            if country_code != target_country:
                question = question.replace(f"in {country_name}", f"in {COUNTRY_NAMES[target_country]}")
                question = question.replace(country_name, COUNTRY_NAMES[target_country])

        # Replace "Deutschland" specifically
        if "Deutschland" in question and target_country != "DE":
            question = question.replace("Deutschland", COUNTRY_NAMES[target_country])

        return question

    def _adapt_answer(
        self,
        question: str,
        original_answer: str,
        target_country: str,
        context: str
    ) -> Optional[str]:
        """Adapt answer to target country using LLM."""

        system_prompt = f"""Du bist ein Experte für Buchhaltung in {COUNTRY_NAMES[target_country]}.

Deine Aufgabe ist es, eine bestehende Buchhaltungsantwort für {COUNTRY_NAMES[target_country]} anzupassen.

WICHTIGE REGELN:
- Behalte die GLEICHE STRUKTUR wie die Original-Antwort bei
- Ersetze Gesetzesverweise durch die entsprechenden Vorschriften in {COUNTRY_NAMES[target_country]}
- Passe Beträge, Steuersätze und Schwellenwerte an
- Verwende die gleiche vorsichtige Sprache ("grundsätzlich", "in der Regel", etc.)
- Behalte Disclaimer bei
- Ändere NICHT den grundlegenden Inhalt oder die Struktur der Antwort

WICHTIG: Generiere NUR die angepasste Antwort, KEINE zusätzlichen Erklärungen."""

        user_prompt = f"""Frage: {question}

Original-Antwort (für ein anderes Land):
{original_answer}

Länderspezifische Informationen für {COUNTRY_NAMES[target_country]}:
{context if context else "(keine spezifischen Regeln verfügbar)"}

Passe die Antwort für {COUNTRY_NAMES[target_country]} an. Behalte die gleiche Struktur bei:"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1500
            )

            answer = response.choices[0].message.content
            return answer.strip() if answer else None

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(samples: List[Dict[str, Any]], path: Path, append: bool = False):
    """Save samples as JSONL."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Expand instruction samples to country variants")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/instruction/gold.jsonl"),
        help="Input JSONL file with instruction samples"
    )
    parser.add_argument(
        "--country-rules",
        type=Path,
        default=Path("data/instruction/country_rules.json"),
        help="Path to country rules JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/instruction/gold_expanded.jsonl"),
        help="Output JSONL file"
    )
    parser.add_argument(
        "--countries",
        type=str,
        default="AT,DE,CH",
        help="Comma-separated list of target countries (AT,DE,CH)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Azure OpenAI deployment name (overrides env var)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature for adaptation (lower = more conservative)"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file"
    )

    args = parser.parse_args()

    # Validate environment
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = args.model or os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not all([endpoint, api_key, deployment]):
        logger.error("Missing Azure OpenAI configuration. Set environment variables:")
        logger.error("  AZURE_OPENAI_ENDPOINT")
        logger.error("  AZURE_OPENAI_API_KEY")
        logger.error("  AZURE_OPENAI_DEPLOYMENT")
        sys.exit(1)

    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    )

    # Load data
    logger.info(f"Loading samples from {args.input}")
    samples = load_jsonl(args.input)
    logger.info(f"Loaded {len(samples)} samples")

    logger.info(f"Loading country rules from {args.country_rules}")
    country_rules = load_json(args.country_rules)

    # Parse countries
    countries = [c.strip() for c in args.countries.split(",")]
    logger.info(f"Target countries: {countries}")

    # Initialize expander
    expander = CountryVariantExpander(
        client=client,
        deployment=deployment,
        country_rules=country_rules,
        temperature=args.temperature
    )

    # Expand samples
    logger.info("Expanding samples to country variants...")
    expanded_samples = []
    total_expansions = len(samples) * len(countries)
    current = 0

    for sample in samples:
        for country in countries:
            current += 1
            logger.info(f"Progress: {current}/{total_expansions} ({100*current/total_expansions:.1f}%) - {sample.get('id', 'unknown')} → {country}")

            expanded = expander.expand_sample(sample, country)
            if expanded:
                expanded_samples.append(expanded)

    # Save expanded samples
    logger.info(f"Saving {len(expanded_samples)} expanded samples to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(expanded_samples, args.output, append=args.append)

    # Statistics
    logger.info("✅ Expansion complete!")
    logger.info(f"Original samples: {len(samples)}")
    logger.info(f"Expanded samples: {len(expanded_samples)}")
    logger.info(f"Expansion factor: {len(expanded_samples)/len(samples):.1f}x")

    country_counts = {}
    for sample in expanded_samples:
        country = sample.get("country", "unknown")
        country_counts[country] = country_counts.get(country, 0) + 1

    logger.info("Samples by country:")
    for country, count in sorted(country_counts.items()):
        logger.info(f"  {country}: {count}")


if __name__ == "__main__":
    main()
