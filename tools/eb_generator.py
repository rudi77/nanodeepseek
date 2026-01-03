"""EB synthetic dataset generator (SFT + DPO)."""

from __future__ import annotations

import argparse
import json
import math
import random
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

from io_utils import append_jsonl, hash_config
from paraphrase import paraphrase_instruction
from schemas import BookEntryLine, make_bookentry, make_dpo_pair, make_sft_chat, q2, today_iso
from validator import validate_bookentry, validate_sft_row, validate_dpo_row

DEFAULT_SYSTEM_PROMPT = (
    "Du bist Buchhaltungsassistent (AT). "
    "Antworte ausschliesslich mit einem JSON-Objekt im Schema bookentry.v1."
)


def load_case_library(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _log_uniform(rng: random.Random, min_val: float, max_val: float) -> Decimal:
    if min_val <= 0 or max_val <= 0:
        return q2(rng.uniform(min_val, max_val))
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    sample = rng.random() * (log_max - log_min) + log_min
    return q2(math.exp(sample))


def pick_amount(rng: random.Random, model: Dict[str, Any]) -> Decimal:
    min_val = float(model.get("min", 0))
    max_val = float(model.get("max", 0))
    return _log_uniform(rng, min_val, max_val)


def compute_posting(template: Dict[str, Any], amount_display: Decimal) -> Tuple[List[BookEntryLine], Dict[str, Any]]:
    rules = template.get("rules", {})
    booking = template.get("booking", {})
    debit_label = booking.get("debit_label")
    credit_label = booking.get("credit_label")

    vat_handling = rules.get("vat_handling", "none")
    vat_rate = Decimal(str(rules.get("vat_rate", 0)))

    net = amount_display
    gross = amount_display
    if vat_handling == "net_to_gross":
        net = amount_display
        gross = q2(net * (Decimal("1") + vat_rate))

    amount_basis = rules.get("amount_basis", "balance")
    amount = gross if amount_basis == "gross" else amount_display

    lines = [
        BookEntryLine(account_label=debit_label, side="Soll", amount=amount),
        BookEntryLine(account_label=credit_label, side="Haben", amount=amount),
    ]

    meta = {
        "vat": {
            "rate": float(vat_rate),
            "net": float(net),
            "gross": float(gross),
        }
        if vat_handling != "none"
        else None
    }
    return lines, meta


def make_instruction_base(template: Dict[str, Any], datum: str, industry: str, amount: Decimal) -> str:
    desc = template.get("description", "")
    return f"{desc}. Datum: {datum}. Branche: {industry}. Betrag: {amount:.2f}."


def mutate_rejected(rng: random.Random, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    mutated = json.loads(json.dumps(entry))
    lines = mutated.get("lines", [])
    if not lines:
        return mutated, "NO_LINES"

    error_classes = ["SWAP_SIDE", "AMOUNT_TWEAK", "ACCOUNT_LABEL"]
    error_class = rng.choice(error_classes)

    if error_class == "SWAP_SIDE":
        idx = rng.randrange(len(lines))
        side = lines[idx].get("side")
        lines[idx]["side"] = "Haben" if side == "Soll" else "Soll"

    elif error_class == "AMOUNT_TWEAK":
        idx = rng.randrange(len(lines))
        amount = Decimal(str(lines[idx].get("amount", 0)))
        tweak = Decimal("0.01") if rng.random() < 0.5 else Decimal("0.02")
        lines[idx]["amount"] = float(q2(amount + tweak))

    elif error_class == "ACCOUNT_LABEL":
        idx = rng.randrange(len(lines))
        label = lines[idx].get("account_label", "")
        lines[idx]["account_label"] = f"{label} (fehlerhaft)"

    return mutated, error_class


def generate_sft_rows(
    *,
    rng: random.Random,
    templates: List[Dict[str, Any]],
    count: int,
    seed: int,
    source: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _ in range(count):
        template = rng.choice(templates)
        amount_display = pick_amount(rng, template.get("amount_model", {}))
        datum = today_iso()
        industry = rng.choice(template.get("industry_focus", ["all"]))
        base = make_instruction_base(template, datum, industry, amount_display)
        instruction = paraphrase_instruction(base)

        lines, vat_meta = compute_posting(template, amount_display)
        entry = make_bookentry(
            datum=datum,
            industry=industry,
            template_id=template.get("template_id"),
            text=instruction,
            lines=lines,
        )

        meta = {
            "source": source,
            "template_id": template.get("template_id"),
            "industry": industry,
            "seed": seed,
            "amount_display": float(q2(amount_display)),
            "amount_bucket": int(amount_display // Decimal("100")),
        }
        if vat_meta:
            meta.update(vat_meta)

        row = make_sft_chat(DEFAULT_SYSTEM_PROMPT, instruction, json.dumps(entry, ensure_ascii=False), meta)
        rows.append(row)
    return rows


def generate_dpo_rows(rng: random.Random, sft_rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in sft_rows:
        assistant = row["messages"][-1]["content"]
        chosen = json.loads(assistant)
        rejected, error_class = mutate_rejected(rng, chosen)
        meta = {
            "source": row.get("meta", {}).get("source"),
            "template_id": row.get("meta", {}).get("template_id"),
            "industry": row.get("meta", {}).get("industry"),
            "seed": seed,
            "error_class": error_class,
            "amount_bucket": row.get("meta", {}).get("amount_bucket"),
        }
        dpo_row = make_dpo_pair(
            prompt=row["messages"][1]["content"],
            chosen=json.dumps(chosen, ensure_ascii=False),
            rejected=json.dumps(rejected, ensure_ascii=False),
            meta=meta,
        )
        rows.append(dpo_row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-library", default="tools/case_library_eb_100.json")
    parser.add_argument("--out-sft", default="train_sft_eb_1000.jsonl")
    parser.add_argument("--out-dpo", default="train_dpo_eb_1000.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-count", type=int, default=1000)
    parser.add_argument("--dpo-count", type=int, default=1000)
    parser.add_argument("--reject-log", default="rejects_eb.jsonl")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    templates = load_case_library(args.case_library)

    config = {
        "seed": args.seed,
        "sft_count": args.sft_count,
        "dpo_count": args.dpo_count,
        "case_library": str(args.case_library),
    }
    config_hash = hash_config(config)

    sft_rows = generate_sft_rows(
        rng=rng,
        templates=templates,
        count=args.sft_count,
        seed=args.seed,
        source="synthetic_template",
    )

    dpo_rows = generate_dpo_rows(rng, sft_rows, args.seed)
    dpo_rows = dpo_rows[: args.dpo_count]

    if args.validate_only:
        for row in sft_rows:
            ok, errors = validate_sft_row(row)
            if not ok:
                append_jsonl(args.reject_log, {"type": "sft", "row": row, "errors": errors})
        for row in dpo_rows:
            ok, errors = validate_dpo_row(row)
            if not ok:
                append_jsonl(args.reject_log, {"type": "dpo", "row": row, "errors": errors})
        return

    Path(args.out_sft).write_text("", encoding="utf-8")
    Path(args.out_dpo).write_text("", encoding="utf-8")

    for row in sft_rows:
        row.setdefault("meta", {})["config_hash"] = config_hash
        append_jsonl(args.out_sft, row)

    for row in dpo_rows:
        row.setdefault("meta", {})["config_hash"] = config_hash
        append_jsonl(args.out_dpo, row)


if __name__ == "__main__":
    main()
