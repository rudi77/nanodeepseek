"""Mix multiple SFT/DPO JSONL datasets into trainable sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from io_utils import read_jsonl, write_jsonl


def _load_sources(config_path: str) -> Dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _sample_rows(rows: List[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(rows):
        return rows
    return rng.sample(rows, count)


def _normalize_system_prompt(row: Dict[str, Any], system_prompt: str) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    if messages[0].get("role") == "system":
        messages[0]["content"] = system_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tools/mix_config.json")
    parser.add_argument("--out-sft", default="train_sft_all.jsonl")
    parser.add_argument("--out-dpo", default="train_dpo_all.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = _load_sources(args.config)
    rng = random.Random(args.seed)

    sft_rows: List[Dict[str, Any]] = []
    for src in cfg.get("sft_sources", []):
        rows = list(read_jsonl(src["path"]))
        take = int(src.get("count", len(rows)))
        subset = _sample_rows(rows, take, rng)
        for row in subset:
            row.setdefault("meta", {})["source_group"] = src.get("name", "unknown")
            if cfg.get("system_prompt"):
                _normalize_system_prompt(row, cfg["system_prompt"])
        sft_rows.extend(subset)

    dpo_rows: List[Dict[str, Any]] = []
    for src in cfg.get("dpo_sources", []):
        rows = list(read_jsonl(src["path"]))
        take = int(src.get("count", len(rows)))
        subset = _sample_rows(rows, take, rng)
        for row in subset:
            row.setdefault("meta", {})["source_group"] = src.get("name", "unknown")
        dpo_rows.extend(subset)

    rng.shuffle(sft_rows)
    rng.shuffle(dpo_rows)

    write_jsonl(args.out_sft, sft_rows)
    write_jsonl(args.out_dpo, dpo_rows)


if __name__ == "__main__":
    main()
