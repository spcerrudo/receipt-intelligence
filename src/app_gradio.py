from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from PIL import Image

from src.config import get_config
from src.pipeline import ReceiptPipeline


CONFIG = get_config()
PIPELINE: ReceiptPipeline | None = None


def get_pipeline() -> ReceiptPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = ReceiptPipeline(CONFIG)
    return PIPELINE


def _to_review_table(raw_entities: dict[str, Any], corrected_entities: dict[str, Any]) -> pd.DataFrame:
    rows = []
    keys = ["vendor_name", "receipt_number", "date", "time", "total", "subtotal", "tax", "address"]
    for key in keys:
        rows.append(
            {
                "field": key,
                "raw_value": raw_entities.get(key),
                "corrected_value": corrected_entities.get(key),
                "confidence": corrected_entities.get("field_confidence", {}).get(key),
            }
        )
    return pd.DataFrame(rows)


def process_receipt(image: Image.Image) -> tuple[str, dict[str, Any], dict[str, Any], pd.DataFrame]:
    if image is None:
        return "", {}, {}, pd.DataFrame()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = Path(tmp.name)
        image.save(temp_path)

    pipeline = get_pipeline()
    ocr_backend = "bedrock_multimodal" if "bedrock_multimodal" in pipeline.ocr_engines else "paddle"
    result = pipeline.run_receipt(
        temp_path,
        ocr_backend=ocr_backend,
        use_preprocessing=(ocr_backend == "paddle"),
        use_llm=pipeline.llm_client is not None,
    )
    raw_entities = result["extracted_entities"]
    corrected_entities = result["corrected_entities"]
    review_table = _to_review_table(raw_entities, corrected_entities)
    return result["raw_text"], raw_entities, corrected_entities, review_table


def answer_receipt_question(corrected_entities: dict[str, Any], question: str) -> str:
    if not corrected_entities:
        return "Upload a receipt first."
    return get_pipeline().answer_question(corrected_entities, question)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Receipt OCR Demo") as demo:
        gr.Markdown("# Receipt OCR and Entity Extraction")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Receipt")
            raw_text = gr.Textbox(label="Raw OCR Text", lines=16)
        with gr.Row():
            raw_entities = gr.JSON(label="Extracted Entities")
            corrected_entities = gr.JSON(label="Corrected Entities")
        review_table = gr.Dataframe(label="Human Review Table", interactive=True)

        run_button = gr.Button("Process Receipt")
        run_button.click(
            fn=process_receipt,
            inputs=[image_input],
            outputs=[raw_text, raw_entities, corrected_entities, review_table],
        )

        gr.Markdown("## Ask Questions from Structured Fields")
        question = gr.Textbox(label="Question", placeholder="What is the total amount?")
        answer = gr.Textbox(label="Answer")
        ask_button = gr.Button("Ask")
        ask_button.click(
            fn=answer_receipt_question,
            inputs=[corrected_entities, question],
            outputs=[answer],
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
