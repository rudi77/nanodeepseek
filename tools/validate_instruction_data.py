"""Validate instruction training data for quality and compliance.

This script performs automated quality checks on instruction samples:
- Schema compliance
- Disclaimer presence (where required)
- Legal reference format
- Cautious language usage
- Forbidden absolute statements
- Country consistency

Usage:
    python tools/validate_instruction_data.py \\
        --input data/instruction/synthetic.jsonl \\
        --report validation_report.json \\
        --verbose
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Validation rules
REQUIRED_FIELDS = ["id", "type", "source", "topic", "language", "messages", "meta"]
REQUIRED_MESSAGE_ROLES = ["user", "assistant"]
REQUIRED_META_FIELDS = ["difficulty", "instruction_type", "reviewed"]

CAUTIOUS_PHRASES = [
    "grundsätzlich",
    "in der regel",
    "abhängig vom einzelfall",
    "unter bestimmten voraussetzungen",
    "kann",
    "sollte",
    "regelmäßig",
    "üblicherweise"
]

FORBIDDEN_PHRASES = [
    "immer",
    "nie",
    "niemals",
    "definitiv",
    "auf jeden fall",
    "garantiert",
    "mit sicherheit",
    "hundertprozentig"
]

DISCLAIMER_PHRASES = [
    "steuerliche beratung",
    "einzelfallprüfung",
    "ersetzt keine"
]

LEGAL_REFERENCE_PATTERNS = [
    r"§\s*\d+",           # § 15
    r"art\.\s*\d+",       # Art. 29
    r"abs\.\s*\d+",       # Abs. 1
    r"nr\.\s*\d+"         # Nr. 2
]

COUNTRY_CODES = ["AT", "DE", "CH"]


def validate_schema(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate sample schema compliance."""
    errors = []

    # Check required top-level fields
    for field in REQUIRED_FIELDS:
        if field not in sample:
            errors.append(f"Missing required field: {field}")

    # Check messages structure
    if "messages" in sample:
        messages = sample["messages"]
        if not isinstance(messages, list):
            errors.append("'messages' must be a list")
        else:
            roles = [m.get("role") for m in messages]
            for required_role in REQUIRED_MESSAGE_ROLES:
                if required_role not in roles:
                    errors.append(f"Missing message role: {required_role}")

            for idx, msg in enumerate(messages):
                if "role" not in msg:
                    errors.append(f"Message {idx} missing 'role'")
                if "content" not in msg:
                    errors.append(f"Message {idx} missing 'content'")

    # Check meta fields
    if "meta" in sample:
        meta = sample["meta"]
        for field in REQUIRED_META_FIELDS:
            if field not in meta:
                errors.append(f"Missing meta field: {field}")

    return len(errors) == 0, errors


def validate_disclaimer(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if disclaimer is present when required."""
    errors = []

    meta = sample.get("meta", {})
    must_include_disclaimer = meta.get("instruction_type") in ["explanation", "case_assessment", "uncertainty"]

    if not must_include_disclaimer:
        return True, []

    # Get assistant message
    assistant_msg = None
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            assistant_msg = msg.get("content", "").lower()
            break

    if not assistant_msg:
        return True, []

    # Check for disclaimer phrases
    has_disclaimer = any(phrase in assistant_msg for phrase in DISCLAIMER_PHRASES)

    if not has_disclaimer:
        errors.append("Missing disclaimer (e.g., 'Diese Darstellung ersetzt keine steuerliche Beratung.')")

    return len(errors) == 0, errors


def validate_legal_reference(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if legal reference is present when required."""
    errors = []

    meta = sample.get("meta", {})
    must_include_legal_ref = meta.get("contains_legal_reference", False)

    if not must_include_legal_ref:
        return True, []

    # Get assistant message
    assistant_msg = None
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            assistant_msg = msg.get("content", "")
            break

    if not assistant_msg:
        return True, []

    # Check for legal reference patterns
    has_legal_ref = any(
        re.search(pattern, assistant_msg, re.IGNORECASE)
        for pattern in LEGAL_REFERENCE_PATTERNS
    )

    if not has_legal_ref:
        errors.append("Missing legal reference (e.g., '§ 15 Abs. 1 UStG')")

    return len(errors) == 0, errors


def validate_cautious_language(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if cautious language is used."""
    errors = []

    # Get assistant message
    assistant_msg = None
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            assistant_msg = msg.get("content", "").lower()
            break

    if not assistant_msg:
        return True, []

    # Check for at least one cautious phrase
    has_cautious = any(phrase in assistant_msg for phrase in CAUTIOUS_PHRASES)

    if not has_cautious:
        errors.append(f"No cautious language found. Use phrases like: {', '.join(CAUTIOUS_PHRASES[:3])}")

    return len(errors) == 0, errors


def validate_no_absolutes(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check for forbidden absolute statements."""
    errors = []

    # Get assistant message
    assistant_msg = None
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            assistant_msg = msg.get("content", "").lower()
            break

    if not assistant_msg:
        return True, []

    # Check for forbidden phrases
    found_forbidden = [phrase for phrase in FORBIDDEN_PHRASES if phrase in assistant_msg]

    if found_forbidden:
        errors.append(f"Forbidden absolute language found: {', '.join(found_forbidden)}")

    return len(errors) == 0, errors


def validate_country_consistency(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check country-specific consistency."""
    errors = []

    country = sample.get("country")
    if not country:
        return True, []

    if country not in COUNTRY_CODES:
        errors.append(f"Invalid country code: {country}. Must be one of: {COUNTRY_CODES}")

    # Get assistant message
    assistant_msg = None
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            assistant_msg = msg.get("content", "")
            break

    if not assistant_msg:
        return True, []

    # Check for conflicting country references
    # This is a basic check - could be enhanced
    country_indicators = {
        "AT": ["österreich", "ugb", "estg (at)"],
        "DE": ["deutschland", "hgb", "estg (de)"],
        "CH": ["schweiz", "or", "dbg", "mwstg"]
    }

    for other_country, indicators in country_indicators.items():
        if other_country == country:
            continue

        found = [ind for ind in indicators if ind.lower() in assistant_msg.lower()]
        if found:
            errors.append(f"Country mismatch: sample is {country} but mentions {other_country} indicators: {found}")

    return len(errors) == 0, errors


def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Run all validation checks on a sample."""

    checks = {
        "schema_valid": validate_schema(sample),
        "has_disclaimer": validate_disclaimer(sample),
        "has_legal_ref": validate_legal_reference(sample),
        "cautious_language": validate_cautious_language(sample),
        "no_absolutes": validate_no_absolutes(sample),
        "country_consistency": validate_country_consistency(sample),
    }

    all_errors = []
    results = {}

    for check_name, (passed, errors) in checks.items():
        results[check_name] = passed
        all_errors.extend(errors)

    overall_pass = all(results.values())

    return {
        "passed": overall_pass,
        "checks": results,
        "errors": all_errors,
        "sample_id": sample.get("id", "unknown")
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_report(report: Dict[str, Any], path: Path):
    """Save validation report as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Validate instruction training data")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file to validate"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Output validation report JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation results"
    )
    parser.add_argument(
        "--flagged-output",
        type=Path,
        help="Save flagged samples to separate JSONL file"
    )

    args = parser.parse_args()

    # Load samples
    print(f"Loading samples from {args.input}...")
    samples = load_jsonl(args.input)
    print(f"Loaded {len(samples)} samples")

    # Validate all samples
    print("Validating samples...")
    validation_results = []
    flagged_samples = []

    for sample in samples:
        result = validate_sample(sample)
        validation_results.append(result)

        if not result["passed"]:
            flagged_samples.append({
                "sample": sample,
                "validation_result": result
            })

        if args.verbose:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"{status} - {result['sample_id']}")
            if not result["passed"]:
                for error in result["errors"]:
                    print(f"    - {error}")

    # Compute statistics
    total = len(validation_results)
    passed = sum(1 for r in validation_results if r["passed"])
    pass_rate = passed / total if total > 0 else 0

    check_stats = {}
    for check_name in validation_results[0]["checks"].keys():
        check_pass = sum(1 for r in validation_results if r["checks"][check_name])
        check_stats[check_name] = {
            "passed": check_pass,
            "total": total,
            "pass_rate": check_pass / total if total > 0 else 0
        }

    # Collect common issues
    error_counts = {}
    for result in validation_results:
        for error in result["errors"]:
            error_counts[error] = error_counts.get(error, 0) + 1

    common_issues = [
        {"issue": error, "count": count}
        for error, count in sorted(error_counts.items(), key=lambda x: -x[1])
    ]

    # Build report
    report = {
        "total_samples": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pass_rate,
        "check_statistics": check_stats,
        "common_issues": common_issues[:10],  # Top 10
        "flagged_count": len(flagged_samples)
    }

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Passed: {passed} ({100*pass_rate:.1f}%)")
    print(f"Failed: {total - passed} ({100*(1-pass_rate):.1f}%)")
    print()
    print("Check breakdown:")
    for check_name, stats in check_stats.items():
        print(f"  {check_name}: {stats['passed']}/{stats['total']} ({100*stats['pass_rate']:.1f}%)")
    print()
    print(f"Samples flagged for review: {len(flagged_samples)}")

    if common_issues:
        print()
        print("Most common issues:")
        for issue in common_issues[:5]:
            print(f"  [{issue['count']}] {issue['issue']}")

    # Save report
    if args.report:
        print(f"\nSaving report to {args.report}")
        save_report(report, args.report)

    # Save flagged samples
    if args.flagged_output and flagged_samples:
        print(f"Saving {len(flagged_samples)} flagged samples to {args.flagged_output}")
        with open(args.flagged_output, "w", encoding="utf-8") as f:
            for item in flagged_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Exit code based on pass rate
    if pass_rate < 0.95:
        print("\n⚠️  WARNING: Pass rate below 95% - review quality!")
        return 1
    else:
        print("\n✅ Validation complete!")
        return 0


if __name__ == "__main__":
    exit(main())
