"""Validation utilities for schema and accounting checks."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Iterable, List, Tuple

from schemas import ALLOWED_SIDES, BOOKENTRY_SCHEMA_VERSION, TOOL_CALL_VERSION, TOOL_RESPONSE_VERSION, q2


def _err(path: str, msg: str) -> Dict[str, Any]:
    return {"path": path, "error": msg}


def validate_bookentry(entry: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    if entry.get("schema_version") != BOOKENTRY_SCHEMA_VERSION:
        errors.append(_err("schema_version", "missing or wrong schema_version"))

    for key in ("datum", "industry", "template_id", "text", "lines"):
        if key not in entry:
            errors.append(_err(key, "missing"))

    lines = entry.get("lines")
    if not isinstance(lines, list) or not lines:
        errors.append(_err("lines", "must be non-empty list"))
        return False, errors

    total_soll = Decimal("0")
    total_haben = Decimal("0")

    for idx, line in enumerate(lines):
        if not isinstance(line, dict):
            errors.append(_err(f"lines[{idx}]", "line must be object"))
            continue
        for key in ("account_label", "side", "amount"):
            if key not in line:
                errors.append(_err(f"lines[{idx}].{key}", "missing"))

        side = line.get("side")
        if side not in ALLOWED_SIDES:
            errors.append(_err(f"lines[{idx}].side", "side must be Soll or Haben"))
        try:
            amount = q2(line.get("amount"))
        except Exception:
            errors.append(_err(f"lines[{idx}].amount", "amount not numeric"))
            amount = Decimal("0")
        if amount <= 0:
            errors.append(_err(f"lines[{idx}].amount", "amount must be positive"))

        if side == "Soll":
            total_soll += amount
        elif side == "Haben":
            total_haben += amount

    if q2(total_soll) != q2(total_haben):
        errors.append(_err("lines", "Soll and Haben totals must match"))

    return len(errors) == 0, errors


def validate_vat_meta(meta: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    vat = meta.get("vat")
    if not vat:
        return True, errors
    try:
        rate = Decimal(str(vat["rate"]))
        net = q2(vat["net"]) if "net" in vat else None
        gross = q2(vat["gross"]) if "gross" in vat else None
        if net is not None and gross is not None:
            expected = q2(net * (Decimal("1") + rate))
            if expected != gross:
                errors.append(_err("meta.vat", "gross does not match net * (1+rate)"))
    except Exception:
        errors.append(_err("meta.vat", "invalid vat meta"))
    return len(errors) == 0, errors


def validate_sft_row(row: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    if row.get("schema_version") is None:
        errors.append(_err("schema_version", "missing"))
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append(_err("messages", "must contain system, user, assistant"))
        return False, errors

    assistant = messages[-1].get("content") if isinstance(messages[-1], dict) else None
    if not isinstance(assistant, str):
        errors.append(_err("messages[-1].content", "assistant content must be JSON string"))
        return False, errors

    try:
        import json

        entry = json.loads(assistant)
    except Exception:
        errors.append(_err("messages[-1].content", "assistant content not valid JSON"))
        return False, errors

    ok_entry, entry_errors = validate_bookentry(entry)
    errors.extend(entry_errors)

    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    ok_vat, vat_errors = validate_vat_meta(meta)
    errors.extend(vat_errors)

    return len(errors) == 0, errors


def validate_dpo_row(row: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    for key in ("prompt", "chosen", "rejected"):
        if key not in row:
            errors.append(_err(key, "missing"))

    if errors:
        return False, errors

    def _load(label: str, payload: str) -> Tuple[bool, Dict[str, Any] | None]:
        try:
            import json

            return True, json.loads(payload)
        except Exception:
            errors.append(_err(label, "not valid JSON"))
            return False, None

    ok_chosen, chosen = _load("chosen", row.get("chosen", ""))
    ok_rejected, rejected = _load("rejected", row.get("rejected", ""))

    if ok_chosen:
        ok_entry, entry_errors = validate_bookentry(chosen)
        errors.extend(entry_errors)

    if ok_rejected:
        ok_entry, entry_errors = validate_bookentry(rejected)
        # rejected can be wrong, so do not require valid accounting here
        # but still validate schema shape when possible
        for err in entry_errors:
            if err["path"] == "schema_version":
                errors.append(err)

    return len(errors) == 0, errors


def validate_tool_call(tool_call: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Validate a tool call object."""
    errors: List[Dict[str, Any]] = []

    if not isinstance(tool_call, dict):
        errors.append(_err("tool_call", "must be object"))
        return False, errors

    if "tool" not in tool_call:
        errors.append(_err("tool_call.tool", "missing tool name"))

    if "arguments" not in tool_call:
        errors.append(_err("tool_call.arguments", "missing arguments"))
    elif not isinstance(tool_call["arguments"], dict):
        errors.append(_err("tool_call.arguments", "arguments must be object"))

    return len(errors) == 0, errors


def validate_sft_tool_use_row(row: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Validate SFT row with tool use."""
    errors: List[Dict[str, Any]] = []

    if row.get("schema_version") is None:
        errors.append(_err("schema_version", "missing"))

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append(_err("messages", "must contain at least system and user"))
        return False, errors

    # Check for tool calls in assistant messages
    has_tool_calls = False
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(_err(f"messages[{idx}]", "must be object"))
            continue

        role = msg.get("role")
        if role == "assistant" and "tool_calls" in msg:
            has_tool_calls = True
            tool_calls = msg["tool_calls"]
            if not isinstance(tool_calls, list):
                errors.append(_err(f"messages[{idx}].tool_calls", "must be list"))
            else:
                for tc_idx, tc in enumerate(tool_calls):
                    ok_tc, tc_errors = validate_tool_call(tc)
                    for err in tc_errors:
                        errors.append(_err(f"messages[{idx}].tool_calls[{tc_idx}].{err['path']}", err["error"]))

        elif role == "tool":
            # Validate tool response
            content = msg.get("content")
            if not isinstance(content, str):
                errors.append(_err(f"messages[{idx}].content", "tool content must be JSON string"))
            else:
                try:
                    import json
                    tool_resp = json.loads(content)
                    if "tool" not in tool_resp:
                        errors.append(_err(f"messages[{idx}].content.tool", "missing tool name"))
                except Exception:
                    errors.append(_err(f"messages[{idx}].content", "not valid JSON"))

    meta = row.get("meta", {})
    if meta.get("has_tool_calls") and not has_tool_calls:
        errors.append(_err("meta.has_tool_calls", "meta says has_tool_calls but none found"))

    return len(errors) == 0, errors


def validate_dpo_tool_use_row(row: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Validate DPO row with tool use."""
    errors: List[Dict[str, Any]] = []

    for key in ("prompt", "chosen", "rejected"):
        if key not in row:
            errors.append(_err(key, "missing"))

    if errors:
        return False, errors

    def _load_messages(label: str, payload: str) -> Tuple[bool, List[Dict[str, Any]] | None]:
        try:
            import json
            msgs = json.loads(payload)
            if not isinstance(msgs, list):
                errors.append(_err(label, "must be list of messages"))
                return False, None
            return True, msgs
        except Exception:
            errors.append(_err(label, "not valid JSON"))
            return False, None

    ok_chosen, chosen_msgs = _load_messages("chosen", row.get("chosen", ""))
    ok_rejected, rejected_msgs = _load_messages("rejected", row.get("rejected", ""))

    # Validate structure (light validation for rejected as it's expected to be wrong)
    if ok_chosen and chosen_msgs:
        for idx, msg in enumerate(chosen_msgs):
            if not isinstance(msg, dict) or "role" not in msg:
                errors.append(_err(f"chosen[{idx}]", "invalid message structure"))

    return len(errors) == 0, errors


def summarize_validation(results: Iterable[Tuple[bool, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    total = 0
    ok = 0
    reason_counts: Dict[str, int] = {}
    for is_ok, errs in results:
        total += 1
        if is_ok:
            ok += 1
        else:
            for err in errs:
                key = f"{err['path']}::{err['error']}"
                reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "total": total,
        "ok": ok,
        "fail": total - ok,
        "reasons": reason_counts,
    }
