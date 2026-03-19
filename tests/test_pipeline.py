from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.entity_schema import OCRLine
from src.pipeline import ReceiptPipeline


class FakeOCREngine:
    def load_image(self, image_path: str):
        return "image"

    def run(self, image):
        return [
            OCRLine(text="ACME STORE", confidence=0.95, bbox=[]),
            OCRLine(text="123 Market Street", confidence=0.93, bbox=[]),
            OCRLine(text="Receipt No: INV-12345", confidence=0.90, bbox=[]),
            OCRLine(text="Date: 01/02/2024", confidence=0.91, bbox=[]),
            OCRLine(text="Total 12.50", confidence=0.94, bbox=[]),
        ]


def test_run_receipt_returns_expected_fields():
    pipeline = ReceiptPipeline.__new__(ReceiptPipeline)
    pipeline.config = type(
        "Config",
        (),
        {"use_preprocessing": False, "output_root": ROOT / "outputs", "min_field_confidence": 0.65},
    )()
    pipeline.ocr_engines = {"paddle": FakeOCREngine()}
    pipeline.available_ocr_backends = ["paddle"]
    pipeline.llm_client = None

    from src.entity_correction import EntityCorrector
    from src.entity_extraction import EntityExtractor
    from src.evaluation import Evaluator

    pipeline.extractor = EntityExtractor(config=pipeline.config, llm_client=None)
    pipeline.corrector = EntityCorrector(llm_client=None)
    pipeline.evaluator = Evaluator()

    result = pipeline.run_receipt(
        "dummy.jpg",
        ocr_backend="paddle",
        use_preprocessing=False,
        use_llm=False,
        save_artifacts=False,
    )

    assert result["corrected_entities"]["vendor_name"] == "ACME STORE"
    assert result["corrected_entities"]["receipt_number"] == "INV-12345"
    assert result["corrected_entities"]["total"] == "12.50"
    assert result["corrected_entities"]["date"] == "2024-02-01"


def test_evaluate_split_returns_exact_assessment_modes():
    pipeline = ReceiptPipeline.__new__(ReceiptPipeline)
    pipeline.config = type(
        "Config",
        (),
        {
            "use_preprocessing": False,
            "output_root": ROOT / "outputs",
            "min_field_confidence": 0.65,
        },
    )()
    fake_engine = FakeOCREngine()
    pipeline.ocr_engines = {"paddle": fake_engine, "bedrock_multimodal": fake_engine}
    pipeline.available_ocr_backends = ["paddle", "bedrock_multimodal"]
    pipeline.llm_client = None

    from src.entity_correction import EntityCorrector
    from src.entity_extraction import EntityExtractor
    from src.entity_schema import DatasetRecord, ReceiptEntities
    from src.evaluation import Evaluator

    class FakeDatasetLoader:
        def load_split(self, split: str):
            return [
                DatasetRecord(
                    split=split,
                    image_path="dummy.jpg",
                    label_path=None,
                    box_path=None,
                    ground_truth=ReceiptEntities(
                        vendor_name="ACME STORE",
                        receipt_number="INV-12345",
                        date="2024-02-01",
                        total="12.50",
                        address="123 Market Street",
                    ),
                )
            ]

    pipeline.extractor = EntityExtractor(config=pipeline.config, llm_client=None)
    pipeline.corrector = EntityCorrector(llm_client=None)
    pipeline.evaluator = Evaluator()
    pipeline.dataset_loader = FakeDatasetLoader()

    outputs = pipeline.evaluate_split("test")

    assert set(outputs) == {
        "raw_ocr",
        "improved_ocr_multimodal_llm",
        "raw_ocr_entity_llm",
        "improved_ocr_multimodal_llm_entity_llm",
    }
