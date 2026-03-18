# AI / LLM Usage

## Where AI Is Used

The optional LLM layer is used for:

1. Ambiguous extraction
   - vendor name
   - address
   - receipt number when OCR is messy
2. Correction and normalization
   - OCR spelling cleanup
   - date normalization
   - numeric cleanup
   - consistency checks
3. Optional multimodal OCR improvement
   - image-to-text OCR using a Bedrock multimodal model

## Guardrails

- JSON-only outputs
- No unsupported guesses
- Nulls preserved when uncertain
- Correction traces include original value, corrected value, and reason

## Supported Providers

- AWS Bedrock
- OpenAI

The code uses a small provider abstraction so the same extraction and correction prompts work across both backends.

For the assessment comparison, multimodal OCR and entity-analysis LLM are tracked separately so their effect can be measured independently.
