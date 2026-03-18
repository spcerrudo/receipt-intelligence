# Presentation Notes

## Core Pitch

This project is a practical receipt understanding baseline: PaddleOCR for raw OCR, multimodal Bedrock OCR for the improved OCR comparison, deterministic rules for obvious fields, and a constrained LLM layer only where ambiguity is high.

## Likely Reviewer Questions

### Why PaddleOCR?

It is fast to integrate, reliable for receipts, and avoids custom OCR training.

### Why a hybrid approach?

Receipts contain both deterministic patterns and ambiguous text. Rules handle cheap wins; the LLM helps only where it adds value.

### How do you separate OCR improvement from entity improvement?

The evaluation loop measures four explicit modes so OCR improvements and entity-analysis improvements are not conflated.

### How do you reduce hallucination risk?

The LLM sees OCR evidence, returns JSON only, and is instructed to leave unsupported fields null.

### Why support Bedrock?

It is common in enterprise settings, works well when AWS credits are available, and keeps the design provider-agnostic.

### What would you improve next?

- Better receipt-number heuristics
- OCR reading-order cleanup
- Richer error analysis and review workflows
