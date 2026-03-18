from __future__ import annotations

import json
import logging

from src.entity_extraction import LLMClient
from src.entity_schema import CorrectionPayload, FieldTrace, ReceiptEntities
from src.utils import normalize_currency, normalize_date, normalize_whitespace


logger = logging.getLogger(__name__)


class EntityCorrector:
    """Normalize and optionally correct extracted receipt entities."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def correct(self, entities: ReceiptEntities, use_llm: bool = False) -> ReceiptEntities:
        """Apply deterministic normalization and optional LLM corrections."""

        corrected = entities.model_copy(deep=True)
        corrected.correction_notes = []

        self._apply_value(corrected, "vendor_name", normalize_whitespace(corrected.vendor_name), "Whitespace cleanup")
        self._apply_value(corrected, "receipt_number", normalize_whitespace(corrected.receipt_number), "Whitespace cleanup")
        self._apply_value(corrected, "date", normalize_date(corrected.date), "Date normalization")
        self._apply_value(corrected, "time", normalize_whitespace(corrected.time), "Time cleanup")
        self._apply_value(corrected, "total", normalize_currency(corrected.total), "Currency normalization")
        self._apply_value(corrected, "subtotal", normalize_currency(corrected.subtotal), "Currency normalization")
        self._apply_value(corrected, "tax", normalize_currency(corrected.tax), "Currency normalization")
        self._apply_value(corrected, "address", normalize_whitespace(corrected.address), "Whitespace cleanup")

        if use_llm and self.llm_client:
            corrected = self._llm_correct(corrected)
        return corrected

    def _apply_value(self, receipt: ReceiptEntities, field_name: str, new_value: str | None, reason: str) -> None:
        old_value = getattr(receipt, field_name)
        if old_value == new_value:
            return
        setattr(receipt, field_name, new_value)
        receipt.correction_notes.append(
            FieldTrace(
                field_name=field_name,
                original_value=old_value,
                corrected_value=new_value,
                reason=reason,
            )
        )

    def _llm_correct(self, entities: ReceiptEntities) -> ReceiptEntities:
        system_prompt = (
            "You normalize receipt entities using OCR-grounded evidence only. "
            "Return JSON only. "
            "Never invent unsupported values. "
            "Preserve null when uncertain. "
            "Include correction_notes with field_name, original_value, corrected_value, and reason."
        )
        user_prompt = json.dumps(
            {
                "task": "Correct OCR-induced mistakes while preserving unsupported fields as null.",
                "raw_text": entities.raw_text,
                "entities": entities.to_evaluation_dict(),
            },
            ensure_ascii=True,
            indent=2,
        )
        try:
            payload = self.llm_client.complete_json(system_prompt, user_prompt)
            parsed = CorrectionPayload.model_validate(payload)
        except Exception as exc:
            logger.warning("LLM correction failed, using deterministic normalization only: %s", exc)
            return entities

        corrected = entities.model_copy(deep=True)
        for field in ["vendor_name", "receipt_number", "date", "time", "total", "subtotal", "tax", "address"]:
            value = getattr(parsed, field)
            if value is not None:
                setattr(corrected, field, value)
        corrected.correction_notes.extend(parsed.correction_notes)
        return corrected
