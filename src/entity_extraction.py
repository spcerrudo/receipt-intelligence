from __future__ import annotations

import json
import logging
import re
from typing import Any

import boto3
from openai import OpenAI

from src.config import AppConfig
from src.entity_schema import ENTITY_FIELDS, OCRLine, ReceiptEntities
from src.utils import normalize_currency, normalize_date, normalize_whitespace


logger = logging.getLogger(__name__)


DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b")
TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
MONEY_PATTERN = re.compile(r"(?<!\d)(\d{1,4}(?:[.,]\d{3})*(?:[.,]\d{2}))(?!\d)")
RECEIPT_PATTERN = re.compile(r"(?i)(?:receipt|invoice|bill|ref|no|number)[^A-Z0-9]{0,5}([A-Z0-9-]{3,})")
ADDRESS_HINTS = ("road", "street", "st", "ave", "jalan", "lorong", "taman", "plaza", "mall")


class LLMClient:
    """JSON-only helper for optional extraction/correction calls."""

    def __init__(self, config: AppConfig) -> None:
        self.provider = config.llm_provider
        self.model = config.openai_model if self.provider == "openai" else config.bedrock_model_id
        self.client: Any

        if self.provider == "openai":
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not configured")
            kwargs: dict[str, Any] = {"api_key": config.openai_api_key}
            if config.openai_base_url:
                kwargs["base_url"] = config.openai_base_url
            self.client = OpenAI(**kwargs)
            return

        if self.provider == "bedrock":
            self.client = boto3.client("bedrock-runtime", region_name=config.aws_region)
            return

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Request a strict JSON object from the LLM."""

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)

        response = self.client.converse(
            modelId=self.model,
            inferenceConfig={"temperature": 0},
            system=[{"text": f"{system_prompt} Return a valid JSON object only."}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
        )
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        content = "".join(block.get("text", "") for block in content_blocks) or "{}"
        content = self._extract_json_object(content)
        return json.loads(content)

    def _extract_json_object(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in LLM response")
        return text[start : end + 1]


class EntityExtractor:
    """Hybrid entity extraction using rules first, optional LLM second."""

    def __init__(self, config: AppConfig, llm_client: LLMClient | None = None) -> None:
        self.config = config
        self.llm_client = llm_client

    def extract(self, ocr_lines: list[OCRLine], use_llm: bool = False) -> ReceiptEntities:
        """Extract entities from OCR lines."""

        raw_text = "\n".join(line.text for line in ocr_lines if line.text)
        entities = ReceiptEntities(raw_text=raw_text)
        entities.field_confidence = self._rule_confidence(ocr_lines)

        lines = [normalize_whitespace(line.text) for line in ocr_lines if line.text.strip()]
        entities.vendor_name = self._extract_vendor(lines)
        entities.date = normalize_date(self._search_first(DATE_PATTERN, raw_text))
        entities.time = self._search_first(TIME_PATTERN, raw_text)
        entities.receipt_number = self._extract_receipt_number(lines, raw_text)
        totals = self._extract_money_fields(lines)
        entities.total = totals.get("total")
        entities.subtotal = totals.get("subtotal")
        entities.tax = totals.get("tax")
        entities.address = self._extract_address(lines)

        if use_llm and self.llm_client:
            entities = self._merge_llm_extraction(entities, lines)
        return entities

    def _rule_confidence(self, ocr_lines: list[OCRLine]) -> dict[str, float]:
        if not ocr_lines:
            return {}
        avg_conf = sum(line.confidence for line in ocr_lines) / len(ocr_lines)
        return {field: avg_conf for field in ENTITY_FIELDS}

    def _search_first(self, pattern: re.Pattern[str], text: str) -> str | None:
        match = pattern.search(text)
        return match.group(0) if match else None

    def _extract_vendor(self, lines: list[str | None]) -> str | None:
        for line in lines[:5]:
            if not line or len(line) < 3:
                continue
            if re.search(r"\d{2,}", line):
                continue
            if any(keyword in line.lower() for keyword in ["total", "tax", "cash", "change"]):
                continue
            return line
        return None

    def _extract_receipt_number(self, lines: list[str | None], raw_text: str) -> str | None:
        match = RECEIPT_PATTERN.search(raw_text)
        if match:
            return match.group(1)
        for line in lines:
            if not line:
                continue
            if re.search(r"(?i)(receipt|invoice|bill)", line):
                tokens = re.findall(r"[A-Z0-9-]{3,}", line.upper())
                if tokens:
                    return tokens[-1]
        return None

    def _extract_money_fields(self, lines: list[str | None]) -> dict[str, str | None]:
        results: dict[str, str | None] = {"total": None, "subtotal": None, "tax": None}
        for line in lines:
            if not line:
                continue
            amount_match = MONEY_PATTERN.search(line.replace(",", ""))
            if not amount_match:
                continue
            amount = normalize_currency(amount_match.group(1))
            lower = line.lower()
            if "subtotal" in lower and not results["subtotal"]:
                results["subtotal"] = amount
            elif "tax" in lower or "gst" in lower or "vat" in lower:
                results["tax"] = amount
            elif "total" in lower and not results["total"]:
                results["total"] = amount

        if not results["total"]:
            all_amounts = [
                normalize_currency(match.group(1))
                for line in lines
                if line
                for match in MONEY_PATTERN.finditer(line.replace(",", ""))
            ]
            numeric_amounts: list[float] = []
            for value in all_amounts:
                if not value:
                    continue
                try:
                    numeric_amounts.append(float(value))
                except ValueError:
                    continue
            if numeric_amounts:
                results["total"] = f"{max(numeric_amounts):.2f}"
        return results

    def _extract_address(self, lines: list[str | None]) -> str | None:
        address_lines: list[str] = []
        for line in lines[:8]:
            if not line:
                continue
            lower = line.lower()
            if any(hint in lower for hint in ADDRESS_HINTS) or re.search(r"\d{2,}", line):
                address_lines.append(line)
        return ", ".join(address_lines[:3]) if address_lines else None

    def _merge_llm_extraction(self, entities: ReceiptEntities, lines: list[str | None]) -> ReceiptEntities:
        system_prompt = (
            "You extract receipt entities from OCR text. "
            "Return valid JSON only. "
            "Do not invent values. "
            "Use null for unsupported fields."
        )
        user_prompt = json.dumps(
            {
                "task": "Fill only ambiguous receipt fields when grounded in OCR evidence.",
                "ocr_lines": [line for line in lines if line],
                "current_entities": entities.to_evaluation_dict(),
                "expected_fields": ["vendor_name", "receipt_number", "address"],
            },
            ensure_ascii=True,
            indent=2,
        )
        try:
            payload = self.llm_client.complete_json(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning("LLM extraction failed, using rule output only: %s", exc)
            return entities

        for field in ["vendor_name", "receipt_number", "address"]:
            value = payload.get(field)
            if value and not getattr(entities, field):
                setattr(entities, field, normalize_whitespace(str(value)))
        return entities
