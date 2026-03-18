from __future__ import annotations

import logging
import re
from pathlib import Path

from src.entity_schema import DatasetRecord, OCRLine, ReceiptEntities
from src.utils import load_json, read_text, safe_stem


logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".json", ".txt"}


class SROIEDatasetLoader:
    """Load split-aware receipt images and labels from an SROIE-like directory."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)

    def load_split(self, split: str) -> list[DatasetRecord]:
        """Load a dataset split and pair images with labels by filename stem."""

        split_root = self.data_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        image_root = split_root / "img"
        entity_root = split_root / "entities"
        box_root = split_root / "box"

        image_paths = self._collect_files(image_root if image_root.exists() else split_root, IMAGE_EXTENSIONS)
        label_paths = self._collect_files(entity_root if entity_root.exists() else split_root, LABEL_EXTENSIONS)
        box_paths = self._collect_files(box_root if box_root.exists() else split_root, {".txt"})
        label_map = {safe_stem(path): path for path in label_paths}
        box_map = {safe_stem(path): path for path in box_paths}

        records: list[DatasetRecord] = []
        for image_path in sorted(image_paths):
            stem = safe_stem(image_path)
            label_path = label_map.get(stem)
            box_path = box_map.get(stem)
            ground_truth = self.parse_label(label_path) if label_path else ReceiptEntities()
            records.append(
                DatasetRecord(
                    split=split,
                    image_path=str(image_path),
                    label_path=str(label_path) if label_path else None,
                    box_path=str(box_path) if box_path else None,
                    ground_truth=ground_truth,
                )
            )
        logger.info("Loaded %s %s records", len(records), split)
        return records

    def _collect_files(self, root: Path, extensions: set[str]) -> list[Path]:
        return [path for path in root.rglob("*") if path.suffix.lower() in extensions]

    def parse_label(self, path: Path) -> ReceiptEntities:
        """Parse label JSON or text into the common receipt schema."""

        if path.suffix.lower() == ".json":
            payload = load_json(path)
            return self._from_mapping(payload)
        return self._parse_text_label(path)

    def load_box_ocr(self, path: Path | str) -> list[OCRLine]:
        """Parse SROIE box annotations into OCRLine records."""

        lines: list[OCRLine] = []
        for raw_line in read_text(Path(path)).splitlines():
            parsed = self._parse_box_line(raw_line)
            if parsed is not None:
                lines.append(parsed)
        return lines

    def _parse_box_line(self, raw_line: str) -> OCRLine | None:
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 9:
            return None

        coord_parts = parts[:8]
        text = ",".join(parts[8:]).strip()
        try:
            coords = [float(value) for value in coord_parts]
        except ValueError:
            return None

        bbox = [
            [coords[0], coords[1]],
            [coords[2], coords[3]],
            [coords[4], coords[5]],
            [coords[6], coords[7]],
        ]
        return OCRLine(text=text, confidence=1.0, bbox=bbox)

    def _parse_text_label(self, path: Path) -> ReceiptEntities:
        text = read_text(path).strip()
        if not text:
            return ReceiptEntities()

        try:
            payload = load_json(path)
            return self._from_mapping(payload)
        except Exception:
            pass

        mapping: dict[str, str] = {}
        for line in text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                mapping[key.strip()] = value.strip()
        if mapping:
            return self._from_mapping(mapping)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        joined = "\n".join(lines)
        fallback = {
            "raw_text": joined,
            "vendor_name": lines[0] if lines else None,
        }
        total_match = re.search(r"(?i)\btotal\b[^0-9]*([0-9]+(?:[.,][0-9]{2})?)", joined)
        if total_match:
            fallback["total"] = total_match.group(1)
        return self._from_mapping(fallback)

    def _from_mapping(self, payload: dict) -> ReceiptEntities:
        aliases = {
            "company": "vendor_name",
            "vendor": "vendor_name",
            "seller": "vendor_name",
            "invoice_no": "receipt_number",
            "receipt_no": "receipt_number",
            "receipt_num": "receipt_number",
            "datetime": "date",
            "amount": "total",
            "grand_total": "total",
        }

        normalized: dict[str, str] = {}
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            canonical = aliases.get(str(key).strip().lower(), str(key).strip().lower())
            normalized[canonical] = value

        return ReceiptEntities(
            vendor_name=self._string(normalized.get("vendor_name")),
            receipt_number=self._string(normalized.get("receipt_number")),
            date=self._string(normalized.get("date")),
            time=self._string(normalized.get("time")),
            total=self._string(normalized.get("total")),
            subtotal=self._string(normalized.get("subtotal")),
            tax=self._string(normalized.get("tax")),
            address=self._string(normalized.get("address")),
            raw_text=self._string(normalized.get("raw_text")) or "",
        )

    def _string(self, value: object) -> str | None:
        if value is None:
            return None
        return str(value).strip() or None
