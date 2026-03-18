from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig, get_config
from src.dataset_loader import SROIEDatasetLoader
from src.entity_correction import EntityCorrector
from src.entity_extraction import EntityExtractor, LLMClient
from src.entity_schema import ReceiptEntities
from src.evaluation import Evaluator
from src.ocr_engine import BedrockMultimodalOCREngine, BoxFileOCREngine, PaddleOCREngine
from src.preprocess import preprocess_image
from src.utils import save_json, setup_logging


logger = logging.getLogger(__name__)


class ReceiptPipeline:
    """End-to-end OCR, extraction, correction, and evaluation pipeline."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.ocr_engines = self._build_ocr_engines()
        self.llm_client = self._build_llm_client()
        self.extractor = EntityExtractor(self.config, llm_client=self.llm_client)
        self.corrector = EntityCorrector(llm_client=self.llm_client)
        self.evaluator = Evaluator()
        self.dataset_loader = SROIEDatasetLoader(self.config.data_root)

    def _build_ocr_engines(self) -> dict[str, object]:
        engines: dict[str, object] = {"paddle": PaddleOCREngine(lang=self.config.ocr_lang)}
        try:
            engines["bedrock_multimodal"] = BedrockMultimodalOCREngine(
                region_name=self.config.aws_region,
                model_id=self.config.bedrock_mm_model_id,
            )
        except Exception as exc:
            logger.warning("Bedrock multimodal OCR unavailable, continuing without it: %s", exc)
        engines["box_reference"] = BoxFileOCREngine()
        return engines

    def _build_llm_client(self) -> LLMClient | None:
        if self.config.llm_provider == "openai" and not self.config.openai_api_key:
            return None
        try:
            return LLMClient(self.config)
        except Exception as exc:
            logger.warning("LLM client unavailable, continuing without LLM: %s", exc)
            return None

    def run_receipt(
        self,
        image_path: str | Path,
        ocr_backend: str = "paddle",
        box_path: str | Path | None = None,
        use_preprocessing: bool | None = None,
        use_llm: bool = False,
        save_artifacts: bool = False,
    ) -> dict[str, Any]:
        """Process a single receipt image and optionally save artifacts."""

        ocr_lines = self._run_ocr(
            image_path=image_path,
            ocr_backend=ocr_backend,
            box_path=box_path,
            use_preprocessing=use_preprocessing,
        )
        extracted = self.extractor.extract(ocr_lines, use_llm=use_llm)
        corrected = self.corrector.correct(extracted, use_llm=use_llm)

        result = {
            "image_path": str(image_path),
            "ocr_backend": ocr_backend,
            "ocr_lines": [line.model_dump() for line in ocr_lines],
            "raw_text": extracted.raw_text,
            "extracted_entities": extracted.model_dump(),
            "corrected_entities": corrected.model_dump(),
        }

        if save_artifacts:
            stem = Path(image_path).stem
            artifact_name = f"{stem}_{ocr_backend}"
            save_json(self.config.output_root / "raw_ocr" / f"{artifact_name}.json", result)
            save_json(
                self.config.output_root / "corrected_entities" / f"{artifact_name}.json",
                corrected.model_dump(),
            )
        return result

    def _run_ocr(
        self,
        image_path: str | Path,
        ocr_backend: str,
        box_path: str | Path | None,
        use_preprocessing: bool | None,
    ) -> list:
        if ocr_backend == "box_reference":
            if not box_path:
                raise ValueError("box_path is required for box_reference OCR")
            return self.ocr_engines["box_reference"].run_from_box_file(box_path)

        if ocr_backend not in self.ocr_engines:
            raise ValueError(f"OCR backend `{ocr_backend}` is not available")

        engine = self.ocr_engines[ocr_backend]
        use_preprocessing = self.config.use_preprocessing if use_preprocessing is None else use_preprocessing
        image = engine.load_image(image_path)
        if ocr_backend == "paddle" and use_preprocessing:
            image = preprocess_image(image)
        return engine.run(image)

    def evaluate_split(
        self,
        split: str = "test",
        limit: int | None = None,
        log_every: int = 10,
    ) -> dict[str, pd.DataFrame]:
        """Evaluate the requested split across all required modes."""

        records = self.dataset_loader.load_split(split)
        if limit is not None:
            records = records[:limit]
        logger.info("Starting evaluation for split `%s` with %s records", split, len(records))
        modes = {
            "raw_ocr": {"ocr_backend": "paddle", "use_preprocessing": False, "use_llm": False},
            "improved_ocr_multimodal_llm": {
                "ocr_backend": "bedrock_multimodal",
                "use_preprocessing": False,
                "use_llm": False,
            },
            "raw_ocr_entity_llm": {"ocr_backend": "paddle", "use_preprocessing": False, "use_llm": True},
            "improved_ocr_multimodal_llm_entity_llm": {
                "ocr_backend": "bedrock_multimodal",
                "use_preprocessing": False,
                "use_llm": True,
            },
        }
        outputs: dict[str, pd.DataFrame] = {}

        for mode_name, options in modes.items():
            if options["ocr_backend"] not in self.ocr_engines:
                logger.warning("Skipping mode `%s` because OCR backend `%s` is unavailable", mode_name, options["ocr_backend"])
                continue
            comparison_rows: list[dict[str, object]] = []
            error_rows: list[dict[str, str]] = []
            logger.info(
                "Evaluating mode `%s` using backend `%s` with entity LLM=%s",
                mode_name,
                options["ocr_backend"],
                options["use_llm"],
            )
            for idx, record in enumerate(records, start=1):
                if idx == 1 or idx % max(log_every, 1) == 0 or idx == len(records):
                    logger.info(
                        "[%s] Processing record %s/%s: %s",
                        mode_name,
                        idx,
                        len(records),
                        Path(record.image_path).name,
                    )
                try:
                    result = self.run_receipt(
                        record.image_path,
                        ocr_backend=options["ocr_backend"],
                        box_path=record.box_path,
                        use_preprocessing=options["use_preprocessing"],
                        use_llm=options["use_llm"],
                        save_artifacts=False,
                    )
                except Exception as exc:
                    logger.exception("Mode `%s` failed on image `%s`: %s", mode_name, record.image_path, exc)
                    error_rows.append(
                        {
                            "mode": mode_name,
                            "image_path": record.image_path,
                            "error": str(exc),
                        }
                    )
                    continue
                prediction = ReceiptEntities.model_validate(result["corrected_entities"])
                rows = self.evaluator.compare(prediction, record.ground_truth)
                for row in rows:
                    row.update({"image_path": record.image_path, "mode": mode_name})
                comparison_rows.extend(rows)

            if error_rows:
                error_frame = pd.DataFrame(error_rows)
                error_path = self.config.output_root / "metrics" / f"{mode_name}_errors.csv"
                error_frame.to_csv(error_path, index=False)
                logger.warning(
                    "Mode `%s` completed with %s failed records. Errors written to %s",
                    mode_name,
                    len(error_rows),
                    error_path,
                )
            if not comparison_rows:
                logger.warning("Skipping summary write for mode `%s` because no successful records were evaluated", mode_name)
                continue
            details = pd.DataFrame(comparison_rows)
            summary = self.evaluator.summarize(comparison_rows)
            self.evaluator.save_summary(self.config.output_root / "metrics", mode_name, summary, details)
            outputs[mode_name] = summary
            logger.info("\nMode: %s\n%s", mode_name, summary.to_string(index=False))
        return outputs

    def answer_question(self, corrected_entities: dict[str, Any], question: str) -> str:
        """Answer receipt questions from structured fields only."""

        receipt = ReceiptEntities.model_validate(corrected_entities)
        q = question.lower().strip()
        uncertainty = [
            field
            for field, confidence in receipt.field_confidence.items()
            if confidence < self.config.min_field_confidence
        ]

        if "total" in q:
            return receipt.total or "Total is not confidently available."
        if "vendor" in q or "seller" in q:
            return receipt.vendor_name or "Vendor is not confidently available."
        if "date" in q:
            return receipt.date or "Date is not confidently available."
        if "address" in q:
            return receipt.address or "Address is not confidently available."
        if "uncertain" in q or "confidence" in q:
            return ", ".join(uncertainty) if uncertainty else "No fields are currently flagged as uncertain."

        available = {key: value for key, value in receipt.to_evaluation_dict().items() if value}
        return f"Supported fields: {available}" if available else "No structured fields are available."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Receipt OCR pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_one = subparsers.add_parser("run-one", help="Run OCR and extraction for one image")
    run_one.add_argument("--image", required=True, help="Path to a receipt image")
    run_one.add_argument(
        "--ocr-backend",
        default="paddle",
        choices=["paddle", "bedrock_multimodal"],
        help="OCR backend to use",
    )
    run_one.add_argument("--use-preprocessing", action="store_true", help="Enable preprocessing")
    run_one.add_argument("--use-llm", action="store_true", help="Enable LLM extraction/correction")
    run_one.add_argument("--save-artifacts", action="store_true", help="Write OCR and entity outputs to disk")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a dataset split")
    evaluate.add_argument("--split", default="test", help="Dataset split to evaluate")
    evaluate.add_argument("--limit", type=int, default=None, help="Optional limit on number of records to evaluate")
    evaluate.add_argument("--log-every", type=int, default=10, help="Log progress every N records")

    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    logger.info("CLI command received: %s", args.command)
    if args.command == "evaluate":
        logger.info(
            "Preparing evaluation with split=%s limit=%s log_every=%s",
            args.split,
            args.limit,
            args.log_every,
        )
    elif args.command == "run-one":
        logger.info(
            "Preparing single receipt run for image=%s backend=%s use_llm=%s",
            args.image,
            args.ocr_backend,
            args.use_llm,
        )

    logger.info("Initializing pipeline components...")
    pipeline = ReceiptPipeline()
    logger.info("Pipeline initialized. OCR backends available: %s", ", ".join(sorted(pipeline.ocr_engines)))

    if args.command == "run-one":
        result = pipeline.run_receipt(
            image_path=args.image,
            ocr_backend=args.ocr_backend,
            use_preprocessing=args.use_preprocessing,
            use_llm=args.use_llm,
            save_artifacts=args.save_artifacts,
        )
        logger.info("Corrected entities:\n%s", pd.Series(result["corrected_entities"]).to_string())
    elif args.command == "evaluate":
        pipeline.evaluate_split(split=args.split, limit=args.limit, log_every=args.log_every)


if __name__ == "__main__":
    main()
