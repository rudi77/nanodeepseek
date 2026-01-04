"""Generate synthetic instruction training data for DACH accounting assistant.

This script generates instruction samples using Azure OpenAI, incorporating
country-specific rules for Austria (AT), Germany (DE), and Switzerland (CH).

Usage:
    python tools/generate_instruction_data.py \\
        --templates data/instruction/question_templates.json \\
        --country-rules data/instruction/country_rules.json \\
        --output data/instruction/synthetic.jsonl \\
        --count 100 \\
        --model gpt-4

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
import random
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

INSTRUCTION_SCHEMA_VERSION = "instruction.v1"


class InstructionGenerator:
    """Generates instruction training data with country-specific handling."""

    def __init__(
        self,
        client: AzureOpenAI,
        deployment: str,
        country_rules: List[Dict[str, Any]],
        temperature: float = 0.7,
    ):
        self.client = client
        self.deployment = deployment
        self.country_rules = {r["rule_id"]: r for r in country_rules}
        self.temperature = temperature

    def get_country_context(self, rule_ids: List[str], country: str) -> str:
        """Build country-specific context from rules."""
        context_parts = []
        for rule_id in rule_ids:
            rule = self.country_rules.get(rule_id)
            if not rule:
                continue

            country_data = rule["countries"].get(country)
            if not country_data:
                continue

            context_parts.append(f"**{rule['description']}:**")
            for key, value in country_data.items():
                context_parts.append(f"  - {key}: {value}")
            context_parts.append("")

        return "\n".join(context_parts)

    def generate_sample(
        self,
        template: Dict[str, Any],
        country: str,
        seed: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a single instruction sample."""
        if seed is not None:
            random.seed(seed)

        # Select question template
        question_template = random.choice(template["question_templates"])

        # Build question with country-specific values
        question = self._build_question(question_template, template, country)

        # Build country-specific context for answer
        context = ""
        if template.get("required_rules"):
            context = self.get_country_context(template["required_rules"], country)

        # Generate answer
        answer = self._generate_answer(question, context, country, template)

        if not answer:
            logger.warning(f"Failed to generate answer for template {template['template_id']}")
            return None

        # Build sample
        sample = {
            "id": f"{template['template_id']}_{country}_{seed or 0}",
            "type": "instruction",
            "source": "synthetic_template",
            "topic": template["topic"],
            "language": "de",
            "country": country,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ],
            "meta": {
                "difficulty": template["difficulty"],
                "instruction_type": template["instruction_type"],
                "contains_legal_reference": template["answer_structure"].get("must_include_legal_ref", False),
                "reviewed": False,
                "template_id": template["template_id"],
                "country_specific": template.get("country_specific", False)
            },
            "generation_metadata": {
                "model": self.deployment,
                "temperature": self.temperature,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "seed": seed,
                "schema_version": INSTRUCTION_SCHEMA_VERSION
            }
        }

        return sample

    def _build_question(
        self,
        question_template: str,
        template: Dict[str, Any],
        country: str
    ) -> str:
        """Build question from template with country-specific values."""
        question = question_template

        # Replace country name
        question = question.replace("{country_name}", COUNTRY_NAMES[country])

        # Replace parameters if present
        if "parameters" in template:
            params = template["parameters"]

            # Amount parameter
            if "amount" in params and "{amount}" in question:
                amount = random.choice(params["amount"])
                question = question.replace("{amount}", str(amount))

            # Currency parameter
            if "currency_by_country" in params and "{currency}" in question:
                currency = params["currency_by_country"].get(country, "EUR")
                question = question.replace("{currency}", currency)

        return question

    def _generate_answer(
        self,
        question: str,
        context: str,
        country: str,
        template: Dict[str, Any]
    ) -> Optional[str]:
        """Generate structured answer using LLM."""

        system_prompt = self._build_system_prompt(country, template)
        user_prompt = self._build_user_prompt(question, context, template)

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

    def _build_system_prompt(self, country: str, template: Dict[str, Any]) -> str:
        """Build system prompt for answer generation."""
        return f"""Du bist ein professioneller Buchhalter-Assistent für {COUNTRY_NAMES[country]}.

Deine Aufgabe ist es, Buchhaltungsfragen sachlich, korrekt und vorsichtig zu beantworten.

WICHTIGE REGELN:
- Verwende eine klare Struktur (Kurzantwort → Begründung → Details → Hinweis)
- Nutze vorsichtige Sprache: "grundsätzlich", "in der Regel", "abhängig vom Einzelfall"
- Gib spezifische Gesetzesverweise an (z.B. "§ 15 Abs. 1 UStG")
- Verwende KEINE absoluten Aussagen wie "immer", "nie", "definitiv"
- Behalte einen professionellen, sachlichen Ton bei
- Füge einen Disclaimer hinzu, wenn rechtlich/steuerlich relevant

TON:
- Sachlich
- Ruhig
- Professionell
- Unterstützend"""

    def _build_user_prompt(
        self,
        question: str,
        context: str,
        template: Dict[str, Any]
    ) -> str:
        """Build user prompt with question and context."""

        structure = template["answer_structure"]
        sections = structure.get("sections", [])
        must_include_legal_ref = structure.get("must_include_legal_ref", False)
        must_include_disclaimer = structure.get("must_include_disclaimer", False)

        prompt_parts = [
            f"Frage: {question}",
            ""
        ]

        if context:
            prompt_parts.extend([
                "Relevante rechtliche Informationen:",
                context,
                ""
            ])

        prompt_parts.append("Struktur deiner Antwort:")
        for section in sections:
            prompt_parts.append(f"- {section}")
        prompt_parts.append("")

        if must_include_legal_ref:
            prompt_parts.append("⚠️ WICHTIG: Gib spezifische Gesetzesverweise an (z.B. '§ 15 Abs. 1 UStG').")

        if must_include_disclaimer:
            prompt_parts.append("⚠️ WICHTIG: Füge einen Disclaimer hinzu (z.B. 'Diese Darstellung ersetzt keine steuerliche Beratung.').")

        prompt_parts.append("")
        prompt_parts.append("Generiere jetzt die vollständige Antwort:")

        return "\n".join(prompt_parts)


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
    parser = argparse.ArgumentParser(description="Generate instruction training data")
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("data/instruction/question_templates.json"),
        help="Path to question templates JSON"
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
        default=Path("data/instruction/synthetic.jsonl"),
        help="Output JSONL file"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--countries",
        type=str,
        default="AT,DE,CH",
        help="Comma-separated list of countries (AT,DE,CH)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Azure OpenAI deployment name (overrides env var)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
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
    logger.info(f"Loading templates from {args.templates}")
    templates = load_json(args.templates)

    logger.info(f"Loading country rules from {args.country_rules}")
    country_rules = load_json(args.country_rules)

    # Parse countries
    countries = [c.strip() for c in args.countries.split(",")]

    # Initialize generator
    generator = InstructionGenerator(
        client=client,
        deployment=deployment,
        country_rules=country_rules,
        temperature=args.temperature
    )

    # Generate samples
    logger.info(f"Generating {args.count} samples across countries: {countries}")

    samples = []
    generated_count = 0
    seed_base = args.seed if args.seed is not None else random.randint(0, 1000000)

    while generated_count < args.count:
        # Select template
        template = random.choice(templates)

        # Select country
        if template.get("country_specific", False):
            country = random.choice(countries)
        else:
            # For non-country-specific templates, default to DE
            country = "DE"

        # Generate sample
        seed = seed_base + generated_count
        sample = generator.generate_sample(template, country, seed=seed)

        if sample:
            samples.append(sample)
            generated_count += 1

            if generated_count % 10 == 0:
                logger.info(f"Progress: {generated_count}/{args.count} ({100*generated_count/args.count:.1f}%)")

    # Save samples
    logger.info(f"Saving {len(samples)} samples to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(samples, args.output, append=args.append)

    logger.info("✅ Generation complete!")
    logger.info(f"Samples by topic:")

    topic_counts = {}
    for sample in samples:
        topic = sample["topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    for topic, count in sorted(topic_counts.items()):
        logger.info(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
