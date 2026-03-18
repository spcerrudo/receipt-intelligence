from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import boto3
import cv2
import numpy as np

from src.entity_schema import OCRLine
from src.utils import read_text


class BaseOCREngine(ABC):
    """Shared OCR interface for all backends."""

    @abstractmethod
    def load_image(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk."""

    @abstractmethod
    def run(self, image: np.ndarray | str | Path) -> list[OCRLine]:
        """Run OCR and return normalized line outputs."""


class PaddleOCREngine(BaseOCREngine):
    """Thin wrapper around PaddleOCR for receipt inference."""

    def __init__(self, lang: str = "en", use_angle_cls: bool = True) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        try:
            self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
        except TypeError:
            self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
        except ValueError as exc:
            if "Unknown argument: show_log" not in str(exc):
                raise
            self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def load_image(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk."""

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image

    def run(self, image: np.ndarray | str | Path) -> list[OCRLine]:
        """Run OCR and return normalized line outputs."""

        source = image
        if isinstance(image, (str, Path)):
            source = self.load_image(image)

        try:
            result = self._ocr.ocr(source, cls=True)
        except TypeError as exc:
            if "unexpected keyword argument 'cls'" not in str(exc):
                raise
            result = self._ocr.ocr(source)
        lines: list[OCRLine] = []
        if not result:
            return lines

        for block in result:
            if not block:
                continue
            for item in block:
                bbox, (text, confidence) = item
                lines.append(
                    OCRLine(
                        text=str(text).strip(),
                        confidence=float(confidence),
                        bbox=[[float(v) for v in point] for point in bbox],
                    )
                )
        return lines


class BedrockMultimodalOCREngine(BaseOCREngine):
    """Multimodal OCR using an image-capable Bedrock model."""

    def __init__(self, region_name: str, model_id: str) -> None:
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def load_image(self, image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image

    def run(self, image: np.ndarray | str | Path) -> list[OCRLine]:
        if isinstance(image, (str, Path)):
            source_path = Path(image)
            image_bytes = source_path.read_bytes()
            image_format = source_path.suffix.lower().lstrip(".") or "png"
            if image_format == "jpg":
                image_format = "jpeg"
        else:
            success, encoded = cv2.imencode(".png", image)
            if not success:
                raise ValueError("Failed to encode image for Bedrock multimodal OCR")
            image_bytes = encoded.tobytes()
            image_format = "png"

        prompt = (
            "Perform OCR on this receipt image. "
            "Return JSON only with the shape {\"lines\": [{\"text\": \"...\"}, ...]}. "
            "Preserve reading order. "
            "Do not add explanations."
        )
        response = self.client.converse(
            modelId=self.model_id,
            inferenceConfig={"temperature": 0},
            system=[{"text": "You are an OCR engine. Return a valid JSON object only."}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                        {"text": prompt},
                    ],
                }
            ],
        )
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        content = "".join(block.get("text", "") for block in content_blocks)
        payload = json.loads(self._extract_json_object(content))
        return [
            OCRLine(text=str(item.get("text", "")).strip(), confidence=1.0, bbox=[])
            for item in payload.get("lines", [])
            if str(item.get("text", "")).strip()
        ]

    def _extract_json_object(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in multimodal OCR response")
        return text[start : end + 1]


class BoxFileOCREngine(BaseOCREngine):
    """Reads provided SROIE box annotations as OCR lines."""

    def load_image(self, image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image

    def run(self, image: np.ndarray | str | Path) -> list[OCRLine]:
        raise NotImplementedError("Use `run_from_box_file` for box-based OCR")

    def run_from_box_file(self, box_path: str | Path) -> list[OCRLine]:
        lines: list[OCRLine] = []
        for raw_line in read_text(Path(box_path)).splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 9:
                continue
            try:
                coords = [float(value) for value in parts[:8]]
            except ValueError:
                continue
            text = ",".join(parts[8:]).strip()
            if not text:
                continue
            lines.append(
                OCRLine(
                    text=text,
                    confidence=1.0,
                    bbox=[
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]],
                    ],
                )
            )
        return lines
