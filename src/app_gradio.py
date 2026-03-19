from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from PIL import Image

from src.config import get_config
from src.pipeline import ReceiptPipeline
from src.utils import setup_logging


CONFIG = get_config()
PIPELINE: ReceiptPipeline | None = None
METRICS_DIR = CONFIG.output_root / "metrics"
DETAIL_FILES = [
    "raw_ocr_details.csv",
    "improved_ocr_multimodal_llm_details.csv",
    "raw_ocr_entity_llm_details.csv",
    "improved_ocr_multimodal_llm_entity_llm_details.csv",
]
ERROR_FILES = [
    "raw_ocr_errors.csv",
    "improved_ocr_multimodal_llm_errors.csv",
    "raw_ocr_entity_llm_errors.csv",
    "improved_ocr_multimodal_llm_entity_llm_errors.csv",
]


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
    ocr_backend = "bedrock_multimodal" if "bedrock_multimodal" in pipeline.available_ocr_backends else "paddle"
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


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_evaluation_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = _read_csv_if_exists(METRICS_DIR / "combined_mode_overall_summary.csv")
    field_summary = _read_csv_if_exists(METRICS_DIR / "combined_mode_field_summary.csv")

    detail_frames = []
    for filename in DETAIL_FILES:
        frame = _read_csv_if_exists(METRICS_DIR / filename)
        if not frame.empty:
            detail_frames.append(frame)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    error_frames = []
    for filename in ERROR_FILES:
        frame = _read_csv_if_exists(METRICS_DIR / filename)
        if not frame.empty:
            error_frames.append(frame)
    errors = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()

    exact_plot = pd.DataFrame()
    fuzzy_plot = pd.DataFrame()
    if not field_summary.empty:
        exact_plot = field_summary[["mode", "field", "exact_match_accuracy"]].rename(
            columns={"exact_match_accuracy": "value"}
        )
        fuzzy_plot = field_summary[["mode", "field", "avg_fuzzy_score"]].rename(columns={"avg_fuzzy_score": "value"})

    return overall, field_summary, details, errors, exact_plot, fuzzy_plot


def _to_optional_int(value: float | int | None) -> int | None:
    if value in (None, "", 0):
        return None
    return int(value)


def run_evaluation_ui(split: str, limit: int | None, log_every: int) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline = get_pipeline()
    parsed_limit = _to_optional_int(limit)
    parsed_log_every = _to_optional_int(log_every) or 1
    pipeline.evaluate_split(split=split, limit=parsed_limit, log_every=parsed_log_every)
    overall, field_summary, details, errors, exact_plot, fuzzy_plot = _load_evaluation_artifacts()
    status = f"Evaluation complete for split={split}, limit={parsed_limit or 'all'}."
    return status, overall, field_summary, details, errors, exact_plot, fuzzy_plot


def load_evaluation_ui() -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall, field_summary, details, errors, exact_plot, fuzzy_plot = _load_evaluation_artifacts()
    status = "Loaded existing metrics from outputs/metrics."
    if overall.empty and field_summary.empty and details.empty:
        status = "No evaluation artifacts found yet. Run evaluation first."
    return status, overall, field_summary, details, errors, exact_plot, fuzzy_plot


def filter_detail_rows(details: pd.DataFrame, mode: str, field: str, exact_only: str) -> pd.DataFrame:
    if details is None or len(details) == 0:
        return pd.DataFrame()
    filtered = details.copy()
    if mode != "All":
        filtered = filtered[filtered["mode"] == mode]
    if field != "All":
        filtered = filtered[filtered["field"] == field]
    if exact_only == "Exact Mismatches Only":
        filtered = filtered[filtered["exact_match"] == False]  # noqa: E712
    return filtered.reset_index(drop=True)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Receipt OCR Demo") as demo:
        gr.Markdown("# Receipt OCR and Entity Extraction")
        with gr.Tabs():
            with gr.Tab("Receipt Demo"):
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

            with gr.Tab("Evaluation"):
                gr.Markdown("Run evaluation and inspect the existing metrics, charts, and raw prediction vs ground-truth rows.")
                with gr.Row():
                    split_input = gr.Dropdown(choices=["train", "test"], value="test", label="Split")
                    limit_input = gr.Number(value=5, precision=0, label="Limit")
                    log_every_input = gr.Number(value=1, precision=0, label="Log Every")
                with gr.Row():
                    eval_button = gr.Button("Run Evaluation")
                    load_button = gr.Button("Load Existing Metrics")
                status_text = gr.Textbox(label="Status", interactive=False)

                overall_summary = gr.Dataframe(label="Overall Mode Summary", interactive=False)
                field_summary = gr.Dataframe(label="Per-Field Mode Summary", interactive=False)

                exact_plot = gr.BarPlot(
                    x="field",
                    y="value",
                    color="mode",
                    title="Exact Match Accuracy by Field and Mode",
                    tooltip=["field", "mode", "value"],
                    label="Exact Match Accuracy Chart",
                )
                fuzzy_plot = gr.BarPlot(
                    x="field",
                    y="value",
                    color="mode",
                    title="Average Fuzzy Score by Field and Mode",
                    tooltip=["field", "mode", "value"],
                    label="Fuzzy Score Chart",
                )

                gr.Markdown("### Raw Value Deep Dive")
                with gr.Row():
                    detail_mode = gr.Dropdown(
                        choices=[
                            "All",
                            "raw_ocr",
                            "improved_ocr_multimodal_llm",
                            "raw_ocr_entity_llm",
                            "improved_ocr_multimodal_llm_entity_llm",
                        ],
                        value="All",
                        label="Mode Filter",
                    )
                    detail_field = gr.Dropdown(
                        choices=[
                            "All",
                            "vendor_name",
                            "receipt_number",
                            "date",
                            "time",
                            "total",
                            "subtotal",
                            "tax",
                            "address",
                        ],
                        value="All",
                        label="Field Filter",
                    )
                    mismatch_filter = gr.Dropdown(
                        choices=["All Rows", "Exact Mismatches Only"],
                        value="All Rows",
                        label="Mismatch Filter",
                    )
                detail_rows = gr.Dataframe(label="Raw Predicted vs Ground Truth Rows", interactive=False)
                error_rows = gr.Dataframe(label="Runtime Errors", interactive=False)
                details_state = gr.State(pd.DataFrame())

                eval_outputs = [
                    status_text,
                    overall_summary,
                    field_summary,
                    details_state,
                    error_rows,
                    exact_plot,
                    fuzzy_plot,
                ]

                eval_button.click(
                    fn=run_evaluation_ui,
                    inputs=[split_input, limit_input, log_every_input],
                    outputs=eval_outputs,
                ).then(
                    fn=filter_detail_rows,
                    inputs=[details_state, detail_mode, detail_field, mismatch_filter],
                    outputs=[detail_rows],
                )

                load_button.click(
                    fn=load_evaluation_ui,
                    inputs=[],
                    outputs=eval_outputs,
                ).then(
                    fn=filter_detail_rows,
                    inputs=[details_state, detail_mode, detail_field, mismatch_filter],
                    outputs=[detail_rows],
                )

                for component in [detail_mode, detail_field, mismatch_filter]:
                    component.change(
                        fn=filter_detail_rows,
                        inputs=[details_state, detail_mode, detail_field, mismatch_filter],
                        outputs=[detail_rows],
                    )
    return demo


if __name__ == "__main__":
    setup_logging()
    app = build_app()
    app.launch()
