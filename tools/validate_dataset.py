"""Validate JSONL datasets and emit reject logs + summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from io_utils import append_jsonl, read_jsonl
from validator import summarize_validation, validate_dpo_row, validate_sft_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default=None)
    parser.add_argument("--dpo", default=None)
    parser.add_argument("--reject-log", default="rejects.jsonl")
    parser.add_argument("--summary", default="summary.json")
    args = parser.parse_args()

    results: Dict[str, Any] = {}

    if args.sft:
        sft_results = []
        for row in read_jsonl(args.sft):
            ok, errors = validate_sft_row(row)
            sft_results.append((ok, errors))
            if not ok:
                append_jsonl(args.reject_log, {"type": "sft", "row": row, "errors": errors})
        results["sft"] = summarize_validation(sft_results)

    if args.dpo:
        dpo_results = []
        for row in read_jsonl(args.dpo):
            ok, errors = validate_dpo_row(row)
            dpo_results.append((ok, errors))
            if not ok:
                append_jsonl(args.reject_log, {"type": "dpo", "row": row, "errors": errors})
        results["dpo"] = summarize_validation(dpo_results)

    Path(args.summary).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
