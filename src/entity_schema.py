from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OCRLine(BaseModel):
    """Single OCR line with spatial metadata."""

    text: str
    confidence: float = 0.0
    bbox: list[list[float]] = Field(default_factory=list)


class FieldTrace(BaseModel):
    """Correction trace for one entity field."""

    field_name: str
    original_value: str | None = None
    corrected_value: str | None = None
    reason: str


class ReceiptEntities(BaseModel):
    """Stable schema for extraction, correction, and evaluation."""

    vendor_name: str | None = None
    receipt_number: str | None = None
    date: str | None = None
    time: str | None = None
    total: str | None = None
    subtotal: str | None = None
    tax: str | None = None
    address: str | None = None
    raw_text: str = ""
    field_confidence: dict[str, float] = Field(default_factory=dict)
    correction_notes: list[FieldTrace] = Field(default_factory=list)

    def to_evaluation_dict(self) -> dict[str, str | None]:
        return {
            "vendor_name": self.vendor_name,
            "receipt_number": self.receipt_number,
            "date": self.date,
            "time": self.time,
            "total": self.total,
            "subtotal": self.subtotal,
            "tax": self.tax,
            "address": self.address,
        }


class CorrectionPayload(BaseModel):
    """Strict payload expected back from the LLM correction layer."""

    vendor_name: str | None = None
    receipt_number: str | None = None
    date: str | None = None
    time: str | None = None
    total: str | None = None
    subtotal: str | None = None
    tax: str | None = None
    address: str | None = None
    correction_notes: list[FieldTrace] = Field(default_factory=list)


class DatasetRecord(BaseModel):
    """Single dataset sample used for evaluation."""

    split: str
    image_path: str
    label_path: str | None = None
    box_path: str | None = None
    ground_truth: ReceiptEntities = Field(default_factory=ReceiptEntities)


ENTITY_FIELDS: list[str] = [
    "vendor_name",
    "receipt_number",
    "date",
    "time",
    "total",
    "subtotal",
    "tax",
    "address",
]


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Compatibility helper for Pydantic serialization."""

    return model.model_dump()
