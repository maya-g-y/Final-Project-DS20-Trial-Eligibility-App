
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re

def normalize_text(x: Any) -> str:
    return str(x).lower() if x is not None else ""

def contains_any(haystack: str, needles: List[str]) -> bool:
    h = haystack.lower()
    return any(n.lower() in h for n in needles)

def contains_all(haystack: str, needles: List[str]) -> bool:
    h = haystack.lower()
    return all(n.lower() in h for n in needles)

def rule_screen(patient: Dict[str, Any], criteria: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    """
    Returns:
      decision: 'eligible'|'not_eligible'|'uncertain'
      reasons: list[str]
      missing: list[str]
    """
    reasons: List[str] = []
    missing: List[str] = []

    # Evaluate exclusions first
    for exc in criteria.get("exclusion", []):
        field = exc["field"]
        op = exc["op"]
        value = exc["value"]

        if field not in patient or patient[field] in [None, ""]:
            missing.append(field)
            continue

        field_text = normalize_text(patient[field])

        hit = False
        if op == "contains_any":
            hit = contains_any(field_text, value)
        elif op == "contains_all":
            hit = contains_all(field_text, value)
        elif op == "==":
            hit = (str(patient[field]).lower() == str(value).lower())
        elif op == "not_contains_any":
            hit = False if not contains_any(field_text, value) else True

        if hit:
            reasons.append(f"Exclusion hit: {field} {op} {value}")
            return "not_eligible", reasons, sorted(set(missing))

    # Evaluate inclusions
    for inc in criteria.get("inclusion", []):
        field = inc["field"]
        op = inc["op"]
        value = inc["value"]

        if field not in patient or patient[field] in [None, ""]:
            missing.append(field)
            continue

        if field in ["age", "egfr"]:
            try:
                x = float(patient[field])
            except Exception:
                missing.append(field)
                continue

            if op == ">=" and not (x >= float(value)):
                reasons.append(f"Failed inclusion: {field} must be >= {value}")
                return "not_eligible", reasons, sorted(set(missing))
            if op == "<=" and not (x <= float(value)):
                reasons.append(f"Failed inclusion: {field} must be <= {value}")
                return "not_eligible", reasons, sorted(set(missing))
            if op == ">" and not (x > float(value)):
                reasons.append(f"Failed inclusion: {field} must be > {value}")
                return "not_eligible", reasons, sorted(set(missing))
            if op == "<" and not (x < float(value)):
                reasons.append(f"Failed inclusion: {field} must be < {value}")
                return "not_eligible", reasons, sorted(set(missing))

        else:
            field_text = normalize_text(patient[field])
            ok = True
            if op == "contains_any":
                ok = contains_any(field_text, value)
            elif op == "contains_all":
                ok = contains_all(field_text, value)
            elif op == "==":
                ok = (str(patient[field]).lower() == str(value).lower())
            elif op == "not_contains_any":
                ok = not contains_any(field_text, value)

            if not ok:
                reasons.append(f"Failed inclusion: {field} {op} {value}")
                return "not_eligible", reasons, sorted(set(missing))

    # If we got here, inclusions passed and no exclusion hit.
    # If we still have missing fields, mark uncertain; otherwise eligible.
    if len(missing) > 0:
        reasons.append("Some required info missing in dataset -> uncertain")
        return "uncertain", reasons, sorted(set(missing))

    reasons.append("All checked criteria satisfied")
    return "eligible", reasons, sorted(set(missing))
