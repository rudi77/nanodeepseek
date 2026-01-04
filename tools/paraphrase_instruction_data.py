"""Paraphrase instruction data to increase diversity.

This script takes existing instruction samples and creates paraphrased variants
to increase linguistic diversity while maintaining semantic content, structure,
and professional tone.

Usage:
    python tools/paraphrase_instruction_data.py \
        --input data/instruction/gold_expanded.jsonl \
        --output data/instruction/paraphrased.jsonl \
        --variants 3 \
        --temperature 0.7

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
from typing import Any, Dict, List, Optional, Tuple

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


PARAPHRASE_STRATEGIES = [
    "question_reformulation",  # Rephrase question differently
    "answer_reordering",       # Reorder sections (while keeping structure)
    "synonym_variation",       # Use synonyms and alternate phrasings
    "example_variation",       # Add or vary examples
]


class InstructionParaphraser:
    """Paraphrases instruction samples while maintaining quality."""

    def __init__(
        self,
        client: AzureOpenAI,
        deployment: str,
        temperature: float = 0.7,
    ):
        self.client = client
        self.deployment = deployment
        self.temperature = temperature

    def paraphrase_sample(
        self,
        sample: Dict[str, Any],
        variant_index: int,
        strategy: str = "synonym_variation"
    ) -> Optional[Dict[str, Any]]:
        """Create a paraphrased variant of a sample."""

        # Extract original messages
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

        # Paraphrase question and answer
        paraphrased_question = self._paraphrase_question(user_msg, strategy)
        paraphrased_answer = self._paraphrase_answer(
            assistant_msg,
            sample.get("topic", "allgemein"),
            strategy
        )

        if not paraphrased_question or not paraphrased_answer:
            logger.warning(f"Failed to paraphrase sample {sample.get('id')}")
            return None

        # Build new sample
        new_sample = {
            **sample,
            "id": f"{sample['id']}_p{variant_index}",
            "messages": [
                {"role": "user", "content": paraphrased_question},
                {"role": "assistant", "content": paraphrased_answer}
            ],
            "generation_metadata": {
                "model": self.deployment,
                "temperature": self.temperature,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "source_sample_id": sample.get("id"),
                "paraphrase_strategy": strategy,
                "variant_index": variant_index
            }
        }

        # Update source
        if new_sample.get("source") in ["manual", "manual_expanded"]:
            new_sample["source"] = "paraphrased"
        elif new_sample.get("source") == "synthetic_template":
            new_sample["source"] = "synthetic_paraphrased"

        # Mark as unreviewed
        if "meta" in new_sample:
            new_sample["meta"]["reviewed"] = False

        return new_sample

    def _paraphrase_question(self, question: str, strategy: str) -> Optional[str]:
        """Paraphrase user question."""

        system_prompt = """Du bist ein Experte für deutsche Sprache und Buchhaltung.

Deine Aufgabe ist es, Buchhaltungsfragen umzuformulieren, dabei aber:
- Den GLEICHEN semantischen Inhalt beizubehalten
- Die Professionalität beizubehalten
- Keine zusätzlichen Informationen hinzuzufügen
- Keine Informationen wegzulassen

Generiere NUR die umformulierte Frage, KEINE zusätzlichen Erklärungen."""

        user_prompt = f"""Formuliere diese Buchhaltungsfrage um:

Original: {question}

Umformulierte Frage:"""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=300
            )

            paraphrased = response.choices[0].message.content
            return paraphrased.strip() if paraphrased else None

        except Exception as e:
            logger.error(f"LLM API error (question): {e}")
            return None

    def _paraphrase_answer(
        self,
        answer: str,
        topic: str,
        strategy: str
    ) -> Optional[str]:
        """Paraphrase assistant answer while maintaining structure."""

        system_prompt = """Du bist ein Experte für professionelle Buchhaltungskommunikation.

Deine Aufgabe ist es, Buchhaltungsantworten umzuformulieren, dabei aber:
- Die GLEICHE Struktur beizubehalten (Kurzantwort, Begründung, etc.)
- Den GLEICHEN fachlichen Inhalt beizubehalten
- ALLE Gesetzesverweise beizubehalten
- Die vorsichtige Sprache beizubehalten ("grundsätzlich", "in der Regel", etc.)
- ALLE Disclaimer beizubehalten
- Einen professionellen, sachlichen Ton beizubehalten

Variiere nur:
- Formulierungen und Satzbau
- Synonyme (wo fachlich korrekt)
- Reihenfolge von Unterpunkten (wo sinnvoll)

Generiere NUR die umformulierte Antwort, KEINE zusätzlichen Erklärungen."""

        user_prompt = f"""Formuliere diese Buchhaltungsantwort um:

Thema: {topic}

Original-Antwort:
{answer}

Umformulierte Antwort:"""

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

            paraphrased = response.choices[0].message.content
            return paraphrased.strip() if paraphrased else None

        except Exception as e:
            logger.error(f"LLM API error (answer): {e}")
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


def save_jsonl(samples: List[Dict[str, Any]], path: Path, append: bool = False):
    """Save samples as JSONL."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Paraphrase instruction samples for diversity")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with instruction samples"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file for paraphrased samples"
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=2,
        help="Number of paraphrased variants per sample"
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
        help="LLM temperature for paraphrasing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Fraction of input samples to paraphrase (0.0-1.0)"
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

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)

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

    # Sample fraction
    if args.sample_fraction < 1.0:
        sample_count = int(len(samples) * args.sample_fraction)
        samples = random.sample(samples, sample_count)
        logger.info(f"Sampled {len(samples)} samples ({args.sample_fraction:.0%})")

    # Initialize paraphraser
    paraphraser = InstructionParaphraser(
        client=client,
        deployment=deployment,
        temperature=args.temperature
    )

    # Paraphrase samples
    logger.info(f"Creating {args.variants} paraphrased variants per sample...")
    paraphrased_samples = []
    total_work = len(samples) * args.variants
    current = 0

    for sample in samples:
        for variant_idx in range(args.variants):
            current += 1
            strategy = random.choice(PARAPHRASE_STRATEGIES)

            logger.info(f"Progress: {current}/{total_work} ({100*current/total_work:.1f}%) - {sample.get('id', 'unknown')} variant {variant_idx + 1}")

            paraphrased = paraphraser.paraphrase_sample(sample, variant_idx + 1, strategy)
            if paraphrased:
                paraphrased_samples.append(paraphrased)

    # Save paraphrased samples
    logger.info(f"Saving {len(paraphrased_samples)} paraphrased samples to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(paraphrased_samples, args.output, append=args.append)

    # Statistics
    logger.info("✅ Paraphrasing complete!")
    logger.info(f"Original samples: {len(samples)}")
    logger.info(f"Paraphrased samples: {len(paraphrased_samples)}")
    logger.info(f"Success rate: {100*len(paraphrased_samples)/total_work:.1f}%")

    topic_counts = {}
    for sample in paraphrased_samples:
        topic = sample.get("topic", "unknown")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    logger.info("Samples by topic:")
    for topic, count in sorted(topic_counts.items()):
        logger.info(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
