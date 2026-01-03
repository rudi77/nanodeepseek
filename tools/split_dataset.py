"""Deterministic dataset splitter with leakage control."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from io_utils import read_jsonl, write_jsonl


def leakage_key(row: Dict[str, Any]) -> Tuple[str, Any]:
    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    template_id = meta.get("template_id", "unknown")
    amount_bucket = meta.get("amount_bucket", "unknown")
    return template_id, amount_bucket


def split_groups(groups: List[List[Dict[str, Any]]], ratios: Tuple[float, float, float], seed: int):
    rng = random.Random(seed)
    rng.shuffle(groups)
    total = len(groups)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    train_groups = groups[:n_train]
    val_groups = groups[n_train : n_train + n_val]
    test_groups = groups[n_train + n_val :]
    return train_groups, val_groups, test_groups


def flatten(groups: Iterable[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        rows.extend(group)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-val", required=True)
    parser.add_argument("--out-test", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", default="0.8,0.1,0.1")
    parser.add_argument("--manifest", default="split_manifest.json")
    args = parser.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3:
        raise SystemExit("ratios must be train,val,test")

    grouped: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input):
        grouped[leakage_key(row)].append(row)

    groups = list(grouped.values())
    train_g, val_g, test_g = split_groups(groups, ratios, args.seed)

    train_rows = flatten(train_g)
    val_rows = flatten(val_g)
    test_rows = flatten(test_g)

    write_jsonl(args.out_train, train_rows)
    write_jsonl(args.out_val, val_rows)
    write_jsonl(args.out_test, test_rows)

    manifest = {
        "input": args.input,
        "seed": args.seed,
        "ratios": ratios,
        "groups": len(groups),
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
