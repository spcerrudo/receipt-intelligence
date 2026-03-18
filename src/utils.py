from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for CLI and app usage."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def read_text(path: Path) -> str:
    """Read text from disk with UTF-8 fallback handling."""

    return path.read_text(encoding="utf-8", errors="ignore")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a Python dictionary."""

    return json.loads(read_text(path))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a JSON payload with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def normalize_whitespace(value: str | None) -> str | None:
    """Collapse whitespace while preserving nulls."""

    if value is None:
        return None
    return re.sub(r"\s+", " ", value).strip()


def normalize_currency(value: str | None) -> str | None:
    """Normalize currency-like strings to a simple decimal representation."""

    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.,-]", "", value).strip()
    if not cleaned:
        return None
    if cleaned.count(".") > 1:
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-1]).replace(",", "") + "." + parts[-1]
    else:
        cleaned = cleaned.replace(",", "")
    if cleaned.count(".") > 1:
        return None
    try:
        return f"{float(cleaned):.2f}"
    except ValueError:
        return None


def normalize_date(value: str | None) -> str | None:
    """Normalize common receipt date formats to YYYY-MM-DD when possible."""

    if value is None:
        return None
    candidate = normalize_whitespace(value)
    if not candidate:
        return None

    patterns = [
        (r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", "{0:04d}-{1:02d}-{2:02d}"),
        (r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", None),
    ]

    for pattern, template in patterns:
        match = re.search(pattern, candidate)
        if not match:
            continue
        a, b, c = match.groups()
        if template:
            return template.format(int(a), int(b), int(c))
        year = int(c) if len(c) == 4 else int(f"20{c}")
        return f"{year:04d}-{int(b):02d}-{int(a):02d}"
    return candidate


def normalize_text_field(value: str | None) -> str | None:
    """Lower-friction normalization for text-heavy fields."""

    if value is None:
        return None
    value = normalize_whitespace(value)
    return value.lower() if value else value


def safe_stem(path: str | Path) -> str:
    """Return a file stem for filenames or paths."""

    return Path(path).stem
