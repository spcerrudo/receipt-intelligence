# Architecture Summary

## Overview

The system is a small staged pipeline:

1. `dataset_loader.py` loads receipt images, entity labels, and provided OCR box annotations.
2. `preprocess.py` applies lightweight OpenCV cleanup for PaddleOCR.
3. `ocr_engine.py` supports both PaddleOCR and multimodal Bedrock OCR behind one interface.
4. `entity_extraction.py` uses rules first, then optional JSON-only LLM extraction.
5. `entity_correction.py` normalizes values and records correction traces.
6. `evaluation.py` computes per-field exact and fuzzy metrics.
7. `app_gradio.py` provides upload, review, and field-grounded QA.

## Why This Design

- Practical enough for a take-home.
- Easy to defend: OCR baseline, deterministic rules, constrained LLM.
- Modular, testable, and easy to extend.

## Evaluation Modes

- `raw_ocr`: PaddleOCR on the original image
- `improved_ocr_multimodal_llm`: multimodal Bedrock OCR on the image
- `raw_ocr_entity_llm`: PaddleOCR plus entity extraction/correction LLM
- `improved_ocr_multimodal_llm_entity_llm`: multimodal Bedrock OCR plus entity extraction/correction LLM
