# Challenge 2: Receipt OCR and Entity Extraction

Minimal, presentation-ready Python project for SROIE Dataset v2 receipt OCR, structured entity extraction, optional LLM-based correction, evaluation, and a lightweight Gradio app.

## What This Project Does

- Runs OCR on receipt images with PaddleOCR
- Runs improved OCR with a multimodal Bedrock model
- Extracts structured fields with a hybrid rules + LLM pipeline
- Applies an auditable correction / normalization layer
- Evaluates the exact four assessment configurations against ground truth
- Exposes a small Gradio app for upload, review, and field-based QA

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment variables:

```bash
cp .env.example .env
```

4. If you are using AWS Bedrock, make sure AWS credentials are available through the normal SDK chain, for example via `aws configure`, environment variables, or an attached role.

5. Place the SROIE Dataset v2 under `data/raw/` using this layout:

```text
data/raw/
|-- train/
|   |-- img/
|   |-- box/
|   `-- entities/
|-- test/
|   |-- img/
|   |-- box/
|   `-- entities/
`-- layoutlm-base-uncased/
```

The loader directly supports this structure. `entities` is treated as ground truth, while `box` can be used as a provided OCR reference.

## LLM Configuration

Example Bedrock configuration:

```env
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_MM_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

If you prefer OpenAI later:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

## OCR Backends

- `paddle`: standard OCR baseline
- `bedrock_multimodal`: image-to-text OCR using a multimodal Bedrock model

## Run OCR + Extraction for One Image

```bash
python -m src.pipeline run-one --image path/to/receipt.jpg --ocr-backend paddle --save-artifacts
```

Example using multimodal OCR plus entity analysis:

```bash
python -m src.pipeline run-one --image path/to/receipt.jpg --ocr-backend bedrock_multimodal --use-llm --save-artifacts
```

## Evaluate a Split

```bash
python -m src.pipeline evaluate --split test
```

This runs the exact four assessment modes:

- `raw_ocr`
- `improved_ocr_multimodal_llm`
- `raw_ocr_entity_llm`
- `improved_ocr_multimodal_llm_entity_llm`

Metrics are written to `outputs/metrics/`.

Centralized comparison artifacts are also written to:

- `outputs/metrics/combined_mode_field_summary.csv`
- `outputs/metrics/combined_mode_overall_summary.csv`
- `outputs/metrics/combined_exact_match_pivot.csv`
- `outputs/metrics/combined_fuzzy_score_pivot.csv`
- `outputs/metrics/combined_extracted_count_pivot.csv`

## How Performance Is Compared Programmatically

For each split and each mode, the pipeline:

1. Loads the image and ground-truth entity JSON.
2. Runs the requested OCR backend.
3. Extracts structured entities.
4. Optionally applies the entity-analysis LLM.
5. Normalizes predictions and labels.
6. Computes:
   - number of entities extracted
   - per-field exact match accuracy
   - fuzzy match score for text-heavy fields such as `vendor_name` and `address`
7. Writes detail and summary CSV files per mode.

It also writes centralized comparison tables across all evaluated modes:

- one combined per-field summary
- one combined overall mode-level summary
- one exact-match pivot table
- one fuzzy-score pivot table
- one extracted-count pivot table

## Launch Gradio App

```bash
python -m src.app_gradio
```

The Gradio app includes:

- a receipt upload and Q&A demo
- an evaluation tab to run the benchmark from the UI
- charts for exact-match and fuzzy-score comparisons across modes
- a raw-value drilldown table showing `predicted` vs `ground_truth`

For a detailed explanation of each metric and how to interpret it, see:

- `docs/metrics_guide.md`

## Testing

```bash
pytest
```
