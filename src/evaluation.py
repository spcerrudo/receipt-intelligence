from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz

from src.entity_schema import ENTITY_FIELDS, ReceiptEntities
from src.utils import normalize_currency, normalize_date, normalize_text_field, normalize_whitespace, save_csv


class Evaluator:
    """Compute field-level exact and fuzzy metrics for receipt entities."""

    def normalize_for_field(self, field_name: str, value: str | None) -> str | None:
        if field_name in {"total", "subtotal", "tax"}:
            return normalize_currency(value)
        if field_name == "date":
            return normalize_date(value)
        if field_name in {"vendor_name", "address"}:
            return normalize_text_field(value)
        return normalize_whitespace(value)

    def compare(self, prediction: ReceiptEntities, truth: ReceiptEntities) -> list[dict[str, object]]:
        """Return per-field comparison rows for one receipt."""

        rows: list[dict[str, object]] = []
        for field_name in ENTITY_FIELDS:
            pred_value = self.normalize_for_field(field_name, getattr(prediction, field_name))
            true_value = self.normalize_for_field(field_name, getattr(truth, field_name))
            exact = pred_value == true_value and true_value is not None
            fuzzy = self._fuzzy_score(field_name, pred_value, true_value)
            rows.append(
                {
                    "field": field_name,
                    "predicted": pred_value,
                    "ground_truth": true_value,
                    "exact_match": exact,
                    "fuzzy_score": fuzzy,
                    "extracted": pred_value is not None,
                }
            )
        return rows

    def summarize(self, comparison_rows: list[dict[str, object]]) -> pd.DataFrame:
        """Aggregate per-field results into a summary table."""

        frame = pd.DataFrame(comparison_rows)
        if frame.empty:
            return pd.DataFrame(
                columns=["field", "num_entities_extracted", "exact_match_accuracy", "avg_fuzzy_score"]
            )

        summary = (
            frame.groupby("field")
            .agg(
                num_entities_extracted=("extracted", "sum"),
                exact_match_accuracy=("exact_match", "mean"),
                avg_fuzzy_score=("fuzzy_score", "mean"),
            )
            .reset_index()
        )
        return summary.sort_values("field").reset_index(drop=True)

    def save_summary(self, output_dir, mode_name: str, summary: pd.DataFrame, details: pd.DataFrame) -> None:
        """Persist detail and summary tables for one evaluation mode."""

        save_csv(output_dir / f"{mode_name}_summary.csv", summary)
        save_csv(output_dir / f"{mode_name}_details.csv", details)

    def _fuzzy_score(self, field_name: str, predicted: str | None, truth: str | None) -> float:
        if predicted is None or truth is None:
            return 0.0
        if field_name not in {"vendor_name", "address"}:
            return 100.0 if predicted == truth else 0.0
        return float(fuzz.token_sort_ratio(predicted, truth))
