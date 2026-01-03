"""Validation utilities for schema and accounting checks."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Iterable, List, Tuple

from schemas import ALLOWED_SIDES, BOOKENTRY_SCHEMA_VERSION, q2


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
