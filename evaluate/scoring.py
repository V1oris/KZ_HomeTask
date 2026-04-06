"""Scoring logic for the structured output task.

Provides parsers for each format, field matching, and scoring functions.
"""

import csv
import io
import json
import xml.etree.ElementTree as ET
from collections import defaultdict


# ---------------------------------------------------------------------------
# Format parsers — each returns a dict or None on failure
# ---------------------------------------------------------------------------

def parse_json(text: str) -> dict | None:
    """Try to parse text as JSON. Return dict or None."""
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def parse_yaml(text: str) -> dict | None:
    """Try to parse text as YAML. Return dict or None."""
    try:
        import yaml
        result = yaml.safe_load(text.strip())
        if isinstance(result, dict):
            return result
        return None
    except Exception:
        return None


def parse_xml(text: str) -> dict | None:
    """Parse XML, extract child elements of <record> as dict."""
    try:
        text = text.strip()
        root = ET.fromstring(text)
        result = {}
        for child in root:
            result[child.tag] = child.text if child.text else ""
        if result:
            return result
        return None
    except ET.ParseError:
        return None


def parse_csv(text: str) -> dict | None:
    """Parse CSV (header + data row) into dict."""
    try:
        text = text.strip()
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if len(rows) >= 1:
            return dict(rows[0])
        return None
    except Exception:
        return None


def parse_toml(text: str) -> dict | None:
    """Try to parse text as TOML. Return dict or None."""
    try:
        # Python 3.11+ has tomllib in stdlib
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        result = tomllib.loads(text.strip())
        if isinstance(result, dict):
            return result
        return None
    except Exception:
        return None


PARSERS = {
    "json": parse_json,
    "yaml": parse_yaml,
    "xml": parse_xml,
    "csv": parse_csv,
    "toml": parse_toml,
}


# ---------------------------------------------------------------------------
# Field matching
# ---------------------------------------------------------------------------

def _normalize_value(val) -> str:
    """Normalize a value to string for comparison."""
    if val is None:
        return ""
    return str(val).strip().lower()


def _values_match(predicted, expected) -> bool:
    """Check if predicted value matches expected value."""
    # Try numeric comparison first
    try:
        pred_num = float(str(predicted))
        exp_num = float(str(expected))
        # For integers, exact match
        if isinstance(expected, int):
            return int(pred_num) == exp_num
        # For floats, tolerance of 0.01
        return abs(pred_num - exp_num) < 0.01
    except (ValueError, TypeError):
        pass

    # Fall back to case-insensitive string comparison
    return _normalize_value(predicted) == _normalize_value(expected)


def match_fields(predicted: dict | None, ground_truth: dict) -> float:
    """Return fraction of ground truth fields correctly matched in predicted.

    Args:
        predicted: Parsed output dict, or None if parsing failed.
        ground_truth: Expected field values.

    Returns:
        Float between 0.0 and 1.0.
    """
    if predicted is None or not ground_truth:
        return 0.0

    correct = 0
    total = len(ground_truth)

    for key, expected_val in ground_truth.items():
        if key in predicted and _values_match(predicted[key], expected_val):
            correct += 1

    return correct / total


# ---------------------------------------------------------------------------
# Per-sample and aggregate scoring
# ---------------------------------------------------------------------------

def score_sample(predicted_text: str, ground_truth: dict, format_name: str) -> dict:
    """Score a single prediction.

    Args:
        predicted_text: The model's raw output string.
        ground_truth: Dict of expected field values.
        format_name: The target format ("json", "yaml", etc.).

    Returns:
        Dict with "valid" (0/1), "field_accuracy" (float), "score" (float).
    """
    parser = PARSERS.get(format_name)
    if parser is None:
        return {"valid": 0, "field_accuracy": 0.0, "score": 0.0}

    parsed = parser(predicted_text)
    valid = 1 if parsed is not None else 0
    field_accuracy = match_fields(parsed, ground_truth)
    score = 0.5 * valid + 0.5 * field_accuracy

    return {
        "valid": valid,
        "field_accuracy": field_accuracy,
        "score": score,
    }


def score_all(predictions: list[dict], ground_truths: list[dict]) -> dict:
    """Score all predictions against ground truths.

    Args:
        predictions: List of dicts with "prediction" and "format" keys.
        ground_truths: List of dicts with "fields" and "format" keys.

    Returns:
        Dict with "overall" score, "per_format" breakdown, and "details".
    """
    assert len(predictions) == len(ground_truths), (
        f"Prediction count ({len(predictions)}) != ground truth count ({len(ground_truths)})"
    )

    details = []
    format_scores = defaultdict(list)

    for pred, gt in zip(predictions, ground_truths):
        fmt = gt["format"]
        result = score_sample(pred["prediction"], gt["fields"], fmt)
        result["format"] = fmt
        details.append(result)
        format_scores[fmt].append(result["score"])

    per_format = {}
    for fmt, scores in sorted(format_scores.items()):
        per_format[fmt] = round(sum(scores) / len(scores), 4)

    all_scores = [d["score"] for d in details]
    overall = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    valid_count = sum(d["valid"] for d in details)

    return {
        "overall": overall,
        "per_format": per_format,
        "total_samples": len(details),
        "valid_count": valid_count,
        "valid_ratio": round(valid_count / len(details), 4) if details else 0.0,
        "details": details,
    }
