"""Schema definitions and helpers for synthetic training data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Optional

BOOKENTRY_SCHEMA_VERSION = "bookentry.v1"
SFT_ENVELOPE_VERSION = "sft.chat.v1"
DPO_ENVELOPE_VERSION = "dpo.v1"
TOOL_CALL_VERSION = "tool_call.v1"
TOOL_RESPONSE_VERSION = "tool_response.v1"

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


def make_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Create a tool call object."""
    return {
        "schema_version": TOOL_CALL_VERSION,
        "tool": tool_name,
        "arguments": arguments,
    }


def make_tool_response(tool_name: str, result: Any, success: bool = True, error: Optional[str] = None) -> Dict[str, Any]:
    """Create a tool response object."""
    return {
        "schema_version": TOOL_RESPONSE_VERSION,
        "tool": tool_name,
        "success": success,
        "result": result if success else None,
        "error": error if not success else None,
    }


def make_sft_tool_use_chat(
    system: str,
    messages: List[Dict[str, Any]],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create SFT chat with tool use conversation.

    messages should be a list of message dicts with:
    - role: "user" | "assistant" | "tool"
    - content: str (for user/assistant) or tool response dict (for tool)
    - tool_calls: optional list of tool call dicts (for assistant)
    """
    full_messages = [{"role": "system", "content": system}] + messages
    return {
        "schema_version": SFT_ENVELOPE_VERSION,
        "messages": full_messages,
        "meta": {**meta, "has_tool_calls": True},
    }


def make_dpo_tool_use_pair(
    prompt: str,
    chosen_messages: List[Dict[str, Any]],
    rejected_messages: List[Dict[str, Any]],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Create DPO pair for tool use."""
    import json
    return {
        "schema_version": DPO_ENVELOPE_VERSION,
        "prompt": prompt,
        "chosen": json.dumps(chosen_messages, ensure_ascii=False),
        "rejected": json.dumps(rejected_messages, ensure_ascii=False),
        "meta": {**meta, "has_tool_calls": True},
    }
