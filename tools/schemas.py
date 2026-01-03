"""Schema definitions and helpers for synthetic training data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Optional

BOOKENTRY_SCHEMA_VERSION = "bookentry.v1"
SFT_ENVELOPE_VERSION = "sft.chat.v1"
DPO_ENVELOPE_VERSION = "dpo.v1"

ALLOWED_SIDES = {"Soll", "Haben"}


@dataclass(frozen=True)
class BookEntryLine:
    account_label: str
    side: str
    amount: Decimal
    ekr_code: Optional[str] = None


def q2(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        d = value
    else:
        d = Decimal(str(value))
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def make_bookentry(
    *,
    datum: str,
    industry: str,
    template_id: str,
    text: str,
    lines: Iterable[BookEntryLine],
) -> Dict[str, Any]:
    return {
        "schema_version": BOOKENTRY_SCHEMA_VERSION,
        "datum": datum,
        "industry": industry,
        "template_id": template_id,
        "text": text,
        "lines": [
            {
                "account_label": line.account_label,
                "side": line.side,
                "amount": float(q2(line.amount)),
                **({"ekr_code": line.ekr_code} if line.ekr_code else {}),
            }
            for line in lines
        ],
    }


def make_sft_chat(system: str, user: str, assistant_json: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SFT_ENVELOPE_VERSION,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_json},
        ],
        "meta": meta,
    }


def make_dpo_pair(prompt: str, chosen: str, rejected: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": DPO_ENVELOPE_VERSION,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "meta": meta,
    }


def today_iso() -> str:
    return date.today().isoformat()
