"""Tool-use synthetic dataset generator (SFT + DPO)."""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from io_utils import append_jsonl, hash_config
from paraphrase import paraphrase_instruction
from schemas import (
    make_dpo_tool_use_pair,
    make_sft_tool_use_chat,
    make_tool_call,
    make_tool_response,
    q2,
    today_iso,
)

DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein Buchhaltungsassistent (AT). "
    "Du hast Zugriff auf Tools für Berechnungen, Kontensuche, Datumsberechnungen und USt-Abfragen. "
    "Nutze die Tools wenn nötig und antworte präzise."
)

# Simplified account mapping (SKR03-like)
ACCOUNT_MAP = {
    "Bank": "1200",
    "Kasse": "1000",
    "Forderungen aus Lieferungen und Leistungen": "1400",
    "Verbindlichkeiten aus Lieferungen und Leistungen": "3300",
    "Umsatzsteuer": "1776",
    "Vorsteuer": "1576",
    "Eröffnungsbilanzkonto": "9000",
}

# VAT rates by country and date
VAT_RATES = {
    "AT": {
        "normal": 0.20,
        "reduced": 0.10,
        "special": 0.13,
    },
    "DE": {
        "normal": 0.19,
        "reduced": 0.07,
    },
    "CH": {
        "normal": 0.081,  # 8.1% (2024+)
        "reduced": 0.026,
    },
}


# ============================================================================
# Rule-based tool implementations
# ============================================================================


def execute_calculator(operation: str, expression: str, precision: int = 2) -> Dict[str, Any]:
    """Execute calculator tool (rule-based)."""
    try:
        if operation == "vat_gross":
            # Parse "net_amount * vat_rate" or similar
            parts = expression.replace(" ", "").split("*")
            if len(parts) == 2:
                net = Decimal(parts[0])
                rate = Decimal(parts[1])
                gross = q2(net * (Decimal("1") + rate))
                return {"result": float(gross), "calculation": f"{net} * (1 + {rate}) = {gross}"}

        elif operation == "vat_net":
            # Parse "gross_amount / (1 + vat_rate)"
            if "/" in expression:
                parts = expression.replace(" ", "").split("/")
                gross = Decimal(parts[0])
                # Expect something like "(1+0.20)"
                divisor = eval(parts[1])  # Safe in controlled context
                net = q2(gross / Decimal(str(divisor)))
                return {"result": float(net), "calculation": f"{gross} / {divisor} = {net}"}

        elif operation == "percentage":
            # amount * rate
            parts = expression.replace(" ", "").split("*")
            if len(parts) == 2:
                amount = Decimal(parts[0])
                rate = Decimal(parts[1])
                result = q2(amount * rate)
                return {"result": float(result), "calculation": f"{amount} * {rate} = {result}"}

        elif operation == "round":
            amount = Decimal(expression)
            rounded = q2(amount)
            return {"result": float(rounded), "calculation": f"round({amount}, 2) = {rounded}"}

        elif operation == "calculate":
            # General calculation (safe eval in controlled context)
            result = eval(expression)
            if isinstance(result, (int, float)):
                result = q2(Decimal(str(result)))
                return {"result": float(result), "calculation": f"{expression} = {result}"}

        return {"error": f"Unsupported operation: {operation}"}

    except Exception as e:
        return {"error": str(e)}


def execute_account_lookup(query: str, chart: str = "SKR03", category: str = "alle") -> Dict[str, Any]:
    """Execute account lookup tool (rule-based)."""
    query_lower = query.lower()

    # Simple search in ACCOUNT_MAP
    matches = []
    for name, number in ACCOUNT_MAP.items():
        if query_lower in name.lower() or query == number:
            matches.append({"name": name, "number": number, "chart": chart})

    if matches:
        return {"accounts": matches}
    else:
        return {"accounts": [], "message": f"Keine Konten gefunden für: {query}"}


def execute_date_calculator(base_date: str, operation: str, value: Optional[int] = None) -> Dict[str, Any]:
    """Execute date calculator tool (rule-based)."""
    try:
        dt = datetime.fromisoformat(base_date)

        if operation == "add_days" and value:
            result_dt = dt + timedelta(days=value)
        elif operation == "add_months" and value:
            # Approximate: 30 days per month
            result_dt = dt + timedelta(days=value * 30)
        elif operation == "subtract_days" and value:
            result_dt = dt - timedelta(days=value)
        elif operation == "month_end":
            # Last day of month
            if dt.month == 12:
                result_dt = datetime(dt.year, 12, 31)
            else:
                result_dt = datetime(dt.year, dt.month + 1, 1) - timedelta(days=1)
        elif operation == "quarter_end":
            # End of quarter
            quarter = (dt.month - 1) // 3
            month = (quarter + 1) * 3
            if month == 12:
                result_dt = datetime(dt.year, 12, 31)
            else:
                result_dt = datetime(dt.year, month + 1, 1) - timedelta(days=1)
        elif operation == "year_end":
            result_dt = datetime(dt.year, 12, 31)
        else:
            return {"error": f"Unknown operation: {operation}"}

        return {"result": result_dt.date().isoformat(), "base_date": base_date, "operation": operation}

    except Exception as e:
        return {"error": str(e)}


def execute_vat_lookup(country: str, date_str: str, category: str = "normal") -> Dict[str, Any]:
    """Execute VAT lookup tool (rule-based)."""
    if country not in VAT_RATES:
        return {"error": f"Country {country} not supported"}

    rates = VAT_RATES[country]
    if category not in rates:
        return {"error": f"Category {category} not found for {country}"}

    rate = rates[category]
    return {
        "country": country,
        "date": date_str,
        "category": category,
        "rate": rate,
        "rate_percent": int(rate * 100),
    }


def execute_balance_checker(lines: List[Dict[str, Any]], tolerance: float = 0.01) -> Dict[str, Any]:
    """Execute balance checker tool (rule-based)."""
    soll = Decimal("0")
    haben = Decimal("0")

    for line in lines:
        amount = Decimal(str(line.get("amount", 0)))
        side = line.get("side", "")

        if side == "Soll":
            soll += amount
        elif side == "Haben":
            haben += amount

    difference = abs(soll - haben)
    is_balanced = difference <= Decimal(str(tolerance))

    return {
        "is_balanced": is_balanced,
        "soll_total": float(soll),
        "haben_total": float(haben),
        "difference": float(difference),
    }


# ============================================================================
# Tool executor
# ============================================================================


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Tuple[Any, bool, Optional[str]]:
    """Execute a tool and return (result, success, error)."""
    try:
        if tool_name == "calculator":
            result = execute_calculator(
                operation=arguments.get("operation", "calculate"),
                expression=arguments.get("expression", ""),
                precision=arguments.get("precision", 2),
            )
            if "error" in result:
                return None, False, result["error"]
            return result, True, None

        elif tool_name == "account_lookup":
            result = execute_account_lookup(
                query=arguments.get("query", ""),
                chart=arguments.get("chart", "SKR03"),
                category=arguments.get("category", "alle"),
            )
            return result, True, None

        elif tool_name == "date_calculator":
            result = execute_date_calculator(
                base_date=arguments.get("base_date", ""),
                operation=arguments.get("operation", ""),
                value=arguments.get("value"),
            )
            if "error" in result:
                return None, False, result["error"]
            return result, True, None

        elif tool_name == "vat_lookup":
            result = execute_vat_lookup(
                country=arguments.get("country", "AT"),
                date_str=arguments.get("date", ""),
                category=arguments.get("category", "normal"),
            )
            if "error" in result:
                return None, False, result["error"]
            return result, True, None

        elif tool_name == "balance_checker":
            result = execute_balance_checker(
                lines=arguments.get("lines", []),
                tolerance=arguments.get("tolerance", 0.01),
            )
            return result, True, None

        else:
            return None, False, f"Unknown tool: {tool_name}"

    except Exception as e:
        return None, False, str(e)


# ============================================================================
# Template parameter generation
# ============================================================================


def generate_param_value(rng: random.Random, param_spec: Dict[str, Any]) -> Any:
    """Generate a parameter value based on spec."""
    ptype = param_spec.get("type")

    if ptype == "amount":
        min_val = param_spec.get("min", 100)
        max_val = param_spec.get("max", 10000)
        # Log-uniform distribution
        log_min = 2  # log10(100)
        log_max = 5  # log10(100000)
        val = 10 ** (rng.uniform(log_min, log_max))
        val = max(min_val, min(max_val, val))
        return float(q2(Decimal(str(val))))

    elif ptype == "precise_amount":
        min_val = param_spec.get("min", 100)
        max_val = param_spec.get("max", 10000)
        val = rng.uniform(min_val, max_val)
        # Add random decimals
        val = val + rng.random()
        return round(val, 3)

    elif ptype == "percentage":
        values = param_spec.get("values", [0.02, 0.03, 0.10, 0.20])
        return rng.choice(values)

    elif ptype == "integer":
        values = param_spec.get("values", [7, 10, 14, 30])
        return rng.choice(values)

    elif ptype == "fixed":
        return param_spec.get("value")

    elif ptype == "date":
        # Random date in last 2 years
        days_ago = rng.randint(0, 730)
        dt = datetime.now() - timedelta(days=days_ago)
        return dt.date().isoformat()

    elif ptype == "multi_line":
        count = param_spec.get("count", 4)
        # Generate balanced lines
        amounts = [float(q2(Decimal(str(rng.uniform(100, 5000))))) for _ in range(count)]
        half = count // 2
        lines = []
        for i, amt in enumerate(amounts[:half]):
            lines.append({"side": "Soll", "amount": amt})
        for i, amt in enumerate(amounts[half:]):
            lines.append({"side": "Haben", "amount": amt})
        # Ensure balance
        soll_sum = sum(l["amount"] for l in lines if l["side"] == "Soll")
        haben_sum = sum(l["amount"] for l in lines if l["side"] == "Haben")
        if abs(soll_sum - haben_sum) > 0.01:
            lines[-1]["amount"] = float(q2(Decimal(str(lines[-1]["amount"] + (soll_sum - haben_sum)))))
        return lines

    return None


def fill_template_params(rng: random.Random, template: Dict[str, Any]) -> Dict[str, Any]:
    """Generate all parameter values for a template."""
    input_params = template.get("input_params", {})
    params = {}

    for param_name, param_spec in input_params.items():
        params[param_name] = generate_param_value(rng, param_spec)

    # Add computed params
    if "vat_rate" in params:
        params["vat_rate_percent"] = int(params["vat_rate"] * 100)
    if "discount_rate" in params:
        params["discount_rate_percent"] = int(params["discount_rate"] * 100)

    return params


def format_instruction(template: Dict[str, Any], params: Dict[str, Any]) -> str:
    """Format instruction string with parameters."""
    pattern = template.get("instruction_pattern", "")
    return pattern.format(**params)


# ============================================================================
# Conversation generation
# ============================================================================


def build_tool_arguments(template: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Build tool call arguments from template and params."""
    tool = template["tool_sequence"][0]  # For single-tool templates

    if tool == "calculator":
        operation = "calculate"
        expression = ""

        # Determine operation based on template
        if "vat" in template.get("sub_category", ""):
            if template["expected_output"] == "gross_amount":
                operation = "vat_gross"
                expression = f"{params['net_amount']} * {params['vat_rate']}"
            elif template["expected_output"] == "net_amount":
                operation = "vat_net"
                expression = f"{params['gross_amount']} / (1 + {params['vat_rate']})"
            elif template["expected_output"] == "vat_amount":
                operation = "percentage"
                expression = f"{params['net_amount']} * {params['vat_rate']}"

        elif "rounding" in template.get("sub_category", ""):
            operation = "round"
            expression = str(params["amount"])

        elif "percentage" in template.get("sub_category", ""):
            operation = "percentage"
            expression = f"{params['invoice_amount']} * {params['discount_rate']}"

        return {"operation": operation, "expression": expression, "precision": 2}

    elif tool == "account_lookup":
        return {
            "query": params.get("account_name", ""),
            "chart": "SKR03",
        }

    elif tool == "date_calculator":
        args = {"base_date": params.get("invoice_date") or params.get("date", today_iso())}

        if template["expected_output"] == "due_date":
            args["operation"] = "add_days"
            args["value"] = params.get("days", 30)
        elif template["expected_output"] == "skonto_deadline":
            args["operation"] = "add_days"
            args["value"] = params.get("days", 14)
        elif template["expected_output"] == "month_end":
            args["operation"] = "month_end"
        elif template["expected_output"] == "quarter_end":
            args["operation"] = "quarter_end"
        elif template["expected_output"] == "year_end":
            args["operation"] = "year_end"

        return args

    elif tool == "vat_lookup":
        return {
            "country": params.get("country", "AT"),
            "date": params.get("date", today_iso()),
            "category": params.get("category", "normal"),
        }

    elif tool == "balance_checker":
        return {
            "lines": params.get("lines", []),
            "tolerance": 0.01,
        }

    return {}


def generate_conversation(
    rng: random.Random,
    template: Dict[str, Any],
    params: Dict[str, Any],
    use_paraphrase: bool = True
) -> List[Dict[str, Any]]:
    """Generate a multi-turn conversation with tool calls."""
    instruction = format_instruction(template, params)

    if use_paraphrase:
        try:
            instruction = paraphrase_instruction(instruction)
        except:
            pass  # Fallback to template instruction

    messages = []

    # User message
    messages.append({"role": "user", "content": instruction})

    # For single-tool templates
    if len(template["tool_sequence"]) == 1:
        tool_name = template["tool_sequence"][0]
        arguments = build_tool_arguments(template, params)

        # Assistant calls tool
        tool_call_obj = make_tool_call(tool_name, arguments)
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call_obj],
        })

        # Tool responds
        result, success, error = execute_tool(tool_name, arguments)
        tool_response = make_tool_response(tool_name, result, success, error)
        messages.append({
            "role": "tool",
            "content": json.dumps(tool_response, ensure_ascii=False),
        })

        # Assistant final answer
        if success:
            answer = format_final_answer(template, result)
        else:
            answer = f"Fehler beim Tool-Aufruf: {error}"

        messages.append({
            "role": "assistant",
            "content": answer,
        })

    # For multi-tool templates (simplified: sequential calls)
    else:
        # Not implemented in this version - would need more complex logic
        pass

    return messages


def format_final_answer(template: Dict[str, Any], tool_result: Any) -> str:
    """Format the final answer based on tool result."""
    expected = template.get("expected_output", "")

    if expected == "gross_amount":
        if isinstance(tool_result, dict) and "result" in tool_result:
            return f"Der Bruttobetrag beträgt {tool_result['result']:.2f} EUR."

    elif expected == "net_amount":
        if isinstance(tool_result, dict) and "result" in tool_result:
            return f"Der Nettobetrag beträgt {tool_result['result']:.2f} EUR."

    elif expected == "vat_amount":
        if isinstance(tool_result, dict) and "result" in tool_result:
            return f"Die Umsatzsteuer beträgt {tool_result['result']:.2f} EUR."

    elif expected == "rounded_amount":
        if isinstance(tool_result, dict) and "result" in tool_result:
            return f"Der gerundete Betrag ist {tool_result['result']:.2f} EUR."

    elif expected == "discount_amount":
        if isinstance(tool_result, dict) and "result" in tool_result:
            return f"Der Skontobetrag beträgt {tool_result['result']:.2f} EUR."

    elif expected == "account_number":
        if isinstance(tool_result, dict) and "accounts" in tool_result:
            accounts = tool_result["accounts"]
            if accounts:
                acc = accounts[0]
                return f"Das Konto '{acc['name']}' hat die Nummer {acc['number']}."
            else:
                return "Kein passendes Konto gefunden."

    elif expected in ["due_date", "skonto_deadline", "month_end", "quarter_end", "year_end"]:
        if isinstance(tool_result, dict) and "result" in tool_result:
            return f"Das Datum ist: {tool_result['result']}"

    elif expected == "vat_rate":
        if isinstance(tool_result, dict) and "rate" in tool_result:
            return f"Der USt-Satz beträgt {tool_result['rate_percent']}% ({tool_result['rate']})."

    elif expected == "is_balanced":
        if isinstance(tool_result, dict) and "is_balanced" in tool_result:
            if tool_result["is_balanced"]:
                return f"Die Buchung ist ausgeglichen (Soll: {tool_result['soll_total']:.2f}, Haben: {tool_result['haben_total']:.2f})."
            else:
                return f"Die Buchung ist NICHT ausgeglichen (Differenz: {tool_result['difference']:.2f})."

    # Fallback
    return json.dumps(tool_result, ensure_ascii=False, indent=2)


# ============================================================================
# Dataset generation
# ============================================================================


def load_templates(path: str | Path) -> List[Dict[str, Any]]:
    """Load tool-use templates from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_sft_rows(
    *,
    rng: random.Random,
    templates: List[Dict[str, Any]],
    count: int,
    seed: int,
    source: str,
    use_paraphrase: bool = True,
) -> List[Dict[str, Any]]:
    """Generate SFT rows with tool use."""
    rows = []

    for _ in range(count):
        template = rng.choice(templates)
        params = fill_template_params(rng, template)
        messages = generate_conversation(rng, template, params, use_paraphrase)

        meta = {
            "source": source,
            "template_id": template.get("template_id"),
            "category": template.get("category"),
            "seed": seed,
        }

        row = make_sft_tool_use_chat(DEFAULT_SYSTEM_PROMPT, messages, meta)
        rows.append(row)

    return rows


def mutate_tool_call(rng: random.Random, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Create a wrong tool call for DPO rejection."""
    mutated = json.loads(json.dumps(messages))

    # Find assistant message with tool_calls
    for msg in mutated:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            if tool_calls:
                tc = tool_calls[0]
                args = tc.get("arguments", {})

                error_classes = ["WRONG_TOOL", "WRONG_PARAM", "MISSING_PARAM"]
                error_class = rng.choice(error_classes)

                if error_class == "WRONG_TOOL":
                    # Change tool name
                    wrong_tools = ["wrong_calculator", "invalid_tool", "unknown"]
                    tc["tool"] = rng.choice(wrong_tools)

                elif error_class == "WRONG_PARAM":
                    # Corrupt a parameter
                    if "expression" in args:
                        args["expression"] = "INVALID"
                    elif "query" in args:
                        args["query"] = ""
                    elif "base_date" in args:
                        args["base_date"] = "invalid-date"

                elif error_class == "MISSING_PARAM":
                    # Remove a required param
                    if args:
                        key_to_remove = rng.choice(list(args.keys()))
                        del args[key_to_remove]

                return mutated, error_class

    return mutated, "NO_TOOL_CALL"


def generate_dpo_rows(
    rng: random.Random,
    sft_rows: List[Dict[str, Any]],
    seed: int
) -> List[Dict[str, Any]]:
    """Generate DPO rows from SFT rows."""
    rows = []

    for sft_row in sft_rows:
        messages = sft_row["messages"][1:]  # Skip system
        user_content = messages[0]["content"]

        chosen_messages = messages
        rejected_messages, error_class = mutate_tool_call(rng, messages)

        meta = {
            "source": sft_row.get("meta", {}).get("source"),
            "template_id": sft_row.get("meta", {}).get("template_id"),
            "seed": seed,
            "error_class": error_class,
        }

        dpo_row = make_dpo_tool_use_pair(user_content, chosen_messages, rejected_messages, meta)
        rows.append(dpo_row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", default="tools/case_library_tool_use_100.json")
    parser.add_argument("--out-sft", default="train_sft_tool_use_1000.jsonl")
    parser.add_argument("--out-dpo", default="train_dpo_tool_use_1000.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-count", type=int, default=1000)
    parser.add_argument("--dpo-count", type=int, default=1000)
    parser.add_argument("--no-paraphrase", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    templates = load_templates(args.templates)

    config = {
        "seed": args.seed,
        "sft_count": args.sft_count,
        "dpo_count": args.dpo_count,
        "templates": str(args.templates),
    }
    config_hash = hash_config(config)

    print(f"Generating {args.sft_count} SFT samples...")
    sft_rows = generate_sft_rows(
        rng=rng,
        templates=templates,
        count=args.sft_count,
        seed=args.seed,
        source="synthetic_tool_use",
        use_paraphrase=not args.no_paraphrase,
    )

    print(f"Generating {args.dpo_count} DPO samples...")
    dpo_rows = generate_dpo_rows(rng, sft_rows, args.seed)
    dpo_rows = dpo_rows[:args.dpo_count]

    # Write output
    Path(args.out_sft).write_text("", encoding="utf-8")
    Path(args.out_dpo).write_text("", encoding="utf-8")

    for row in sft_rows:
        row.setdefault("meta", {})["config_hash"] = config_hash
        append_jsonl(args.out_sft, row)

    for row in dpo_rows:
        row.setdefault("meta", {})["config_hash"] = config_hash
        append_jsonl(args.out_dpo, row)

    print(f"✓ Written {len(sft_rows)} SFT rows to {args.out_sft}")
    print(f"✓ Written {len(dpo_rows)} DPO rows to {args.out_dpo}")


if __name__ == "__main__":
    main()
