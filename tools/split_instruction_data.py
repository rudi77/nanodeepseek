"""Split instruction data into train/val/test sets.

This script splits instruction training data into train, validation, and test sets
with proper stratification and leakage control.

Usage:
    python tools/split_instruction_data.py \
        --input data/instruction/all_instruction_data.jsonl \
        --output-dir data/instruction/splits \
        --split 0.8,0.1,0.1 \
        --stratify-by topic \
        --seed 42

Output files:
    - train.jsonl
    - val.jsonl
    - test.jsonl
    - split_manifest.json (metadata about the split)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: List[Dict[str, Any]], path: Path):
    """Save samples as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_json(data: Any, path: Path):
    """Save data as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stratified_split(
    samples: List[Dict[str, Any]],
    split_ratios: Tuple[float, float, float],
    stratify_key: str,
    seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split samples into train/val/test with stratification.

    Args:
        samples: List of samples to split
        split_ratios: Tuple of (train_ratio, val_ratio, test_ratio)
        stratify_key: Key to stratify by (e.g., 'topic', 'difficulty')
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    random.seed(seed)

    train_ratio, val_ratio, test_ratio = split_ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Group samples by stratify key
    groups = defaultdict(list)
    for sample in samples:
        key = sample.get(stratify_key, "unknown")
        groups[key].append(sample)

    train_samples = []
    val_samples = []
    test_samples = []

    # Split each group proportionally
    for key, group_samples in groups.items():
        # Shuffle group
        random.shuffle(group_samples)

        n = len(group_samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_samples.extend(group_samples[:train_end])
        val_samples.extend(group_samples[train_end:val_end])
        test_samples.extend(group_samples[val_end:])

    # Final shuffle
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def check_leakage(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    leakage_keys: List[str]
) -> Dict[str, Any]:
    """
    Check for data leakage between splits.

    Args:
        train_samples: Training samples
        val_samples: Validation samples
        test_samples: Test samples
        leakage_keys: Keys to check for leakage (e.g., ['id', 'template_id'])

    Returns:
        Dictionary with leakage statistics
    """
    leakage_report = {}

    for key in leakage_keys:
        # Extract values from each split
        train_values = set()
        val_values = set()
        test_values = set()

        for sample in train_samples:
            val = sample.get(key)
            if val:
                # For paraphrased samples, extract base ID
                if "_p" in str(val):
                    val = str(val).rsplit("_p", 1)[0]
                train_values.add(val)

        for sample in val_samples:
            val = sample.get(key)
            if val:
                if "_p" in str(val):
                    val = str(val).rsplit("_p", 1)[0]
                val_values.add(val)

        for sample in test_samples:
            val = sample.get(key)
            if val:
                if "_p" in str(val):
                    val = str(val).rsplit("_p", 1)[0]
                test_values.add(val)

        # Check overlaps
        train_val_overlap = train_values & val_values
        train_test_overlap = train_values & test_values
        val_test_overlap = val_values & test_values

        leakage_report[key] = {
            "train_unique": len(train_values),
            "val_unique": len(val_values),
            "test_unique": len(test_values),
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap),
            "has_leakage": len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0
        }

    return leakage_report


def compute_distribution_stats(
    samples: List[Dict[str, Any]],
    keys: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Compute distribution statistics for given keys.

    Args:
        samples: List of samples
        keys: Keys to compute distributions for

    Returns:
        Dictionary mapping keys to value counts
    """
    stats = {}

    for key in keys:
        counts = defaultdict(int)
        for sample in samples:
            value = sample.get(key, "unknown")
            counts[value] += 1
        stats[key] = dict(counts)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Split instruction data into train/val/test sets")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with all instruction samples"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for split files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="0.8,0.1,0.1",
        help="Split ratios as train,val,test (must sum to 1.0)"
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="topic",
        help="Key to stratify split by (e.g., 'topic', 'difficulty', 'country')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--leakage-keys",
        type=str,
        default="id,template_id",
        help="Comma-separated keys to check for data leakage"
    )
    parser.add_argument(
        "--no-test-split",
        action="store_true",
        help="Only create train/val split (no test set)"
    )

    args = parser.parse_args()

    # Parse split ratios
    split_parts = [float(x.strip()) for x in args.split.split(",")]
    if args.no_test_split:
        if len(split_parts) != 2:
            logger.error("For --no-test-split, provide 2 ratios: train,val")
            return 1
        train_ratio, val_ratio = split_parts
        test_ratio = 0.0
        # Normalize
        total = train_ratio + val_ratio
        train_ratio /= total
        val_ratio /= total
    else:
        if len(split_parts) != 3:
            logger.error("Provide 3 split ratios: train,val,test")
            return 1
        train_ratio, val_ratio, test_ratio = split_parts

    split_ratios = (train_ratio, val_ratio, test_ratio)

    if abs(sum(split_ratios) - 1.0) > 1e-6:
        logger.error(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        return 1

    # Parse leakage keys
    leakage_keys = [k.strip() for k in args.leakage_keys.split(",")]

    # Load samples
    logger.info(f"Loading samples from {args.input}")
    samples = load_jsonl(args.input)
    logger.info(f"Loaded {len(samples)} samples")

    # Perform split
    logger.info(f"Splitting with ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    logger.info(f"Stratifying by: {args.stratify_by}")
    logger.info(f"Random seed: {args.seed}")

    train_samples, val_samples, test_samples = stratified_split(
        samples,
        split_ratios,
        args.stratify_by,
        args.seed
    )

    logger.info(f"Split sizes: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # Check leakage
    logger.info(f"Checking for data leakage on keys: {leakage_keys}")
    leakage_report = check_leakage(train_samples, val_samples, test_samples, leakage_keys)

    has_leakage = any(report["has_leakage"] for report in leakage_report.values())
    if has_leakage:
        logger.warning("⚠️  Data leakage detected!")
        for key, report in leakage_report.items():
            if report["has_leakage"]:
                logger.warning(f"  {key}: train-val overlap={report['train_val_overlap']}, "
                             f"train-test overlap={report['train_test_overlap']}, "
                             f"val-test overlap={report['val_test_overlap']}")
    else:
        logger.info("✅ No data leakage detected")

    # Compute distribution statistics
    distribution_keys = [args.stratify_by, "difficulty", "source", "country"]
    train_stats = compute_distribution_stats(train_samples, distribution_keys)
    val_stats = compute_distribution_stats(val_samples, distribution_keys)
    test_stats = compute_distribution_stats(test_samples, distribution_keys)

    # Save splits
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving train split to {args.output_dir / 'train.jsonl'}")
    save_jsonl(train_samples, args.output_dir / "train.jsonl")

    logger.info(f"Saving val split to {args.output_dir / 'val.jsonl'}")
    save_jsonl(val_samples, args.output_dir / "val.jsonl")

    if not args.no_test_split and len(test_samples) > 0:
        logger.info(f"Saving test split to {args.output_dir / 'test.jsonl'}")
        save_jsonl(test_samples, args.output_dir / "test.jsonl")

    # Save manifest
    manifest = {
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        },
        "split_sizes": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        },
        "stratify_by": args.stratify_by,
        "seed": args.seed,
        "leakage_report": leakage_report,
        "distribution_stats": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats
        }
    }

    manifest_path = args.output_dir / "split_manifest.json"
    logger.info(f"Saving split manifest to {manifest_path}")
    save_json(manifest, manifest_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SPLIT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Train: {len(train_samples)} ({100*len(train_samples)/len(samples):.1f}%)")
    logger.info(f"Val: {len(val_samples)} ({100*len(val_samples)/len(samples):.1f}%)")
    logger.info(f"Test: {len(test_samples)} ({100*len(test_samples)/len(samples):.1f}%)")
    logger.info("")

    logger.info(f"Distribution by {args.stratify_by}:")
    all_keys = set(train_stats[args.stratify_by].keys()) | \
               set(val_stats[args.stratify_by].keys()) | \
               set(test_stats[args.stratify_by].keys())

    for key in sorted(all_keys):
        train_count = train_stats[args.stratify_by].get(key, 0)
        val_count = val_stats[args.stratify_by].get(key, 0)
        test_count = test_stats[args.stratify_by].get(key, 0)
        total_count = train_count + val_count + test_count
        logger.info(f"  {key}: train={train_count}, val={val_count}, test={test_count} (total={total_count})")

    logger.info("\n✅ Split complete!")

    return 0


if __name__ == "__main__":
    exit(main())
