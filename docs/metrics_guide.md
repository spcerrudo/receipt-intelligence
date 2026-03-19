# Metrics Guide

This document explains the metrics produced by the evaluation pipeline, why they matter, and how to interpret high or low values.

## Overall

The evaluation outputs are designed to answer two core questions from the challenge:

1. How many entities did the pipeline extract?
2. How correct were those extracted entities?

The pipeline reports this at both:

- field level
- mode level

The four compared modes are:

- `raw_ocr`
- `improved_ocr_multimodal_llm`
- `raw_ocr_entity_llm`
- `improved_ocr_multimodal_llm_entity_llm`

## Per-Field Metrics

These appear in the per-mode summaries and combined field summaries.

### `field`

- What it is: the entity being evaluated, such as `vendor_name`, `date`, `total`, or `address`
- Why it matters: different fields have different difficulty and business importance
- High or low meaning: not itself a score; it is the grouping key used for evaluation

### `num_entities_extracted`

- What it is: the number of times the pipeline produced a non-null value for that field
- Why it matters: measures extraction coverage
- High means:
  - the pipeline is able or willing to extract that field often
  - good field coverage
- Low means:
  - the field is often missed
  - OCR may be weak, heuristics may be weak, or the system may be conservative
- Caveat:
  - high extraction count is only good when paired with acceptable correctness

### `exact_match_accuracy`

- What it is: the fraction of predictions that exactly match normalized ground truth for that field
- Why it matters: strongest correctness metric for structured extraction
- High means:
  - the field is being extracted correctly and consistently
  - especially important for `date`, `receipt_number`, and `total`
- Low means:
  - the field is often wrong, incomplete, noisy, or missing
- Best interpretation:
  - combine it with `num_entities_extracted`
  - high extraction + high exact match is strong
  - high extraction + low exact match means the system may be overpredicting

### `avg_fuzzy_score`

- What it is: the average fuzzy similarity between predicted and true text
- Why it matters:
  - exact match is too strict for text-heavy fields such as `vendor_name` and `address`
  - fuzzy score captures partially correct outputs
- High means:
  - the predicted value is close to the ground truth
  - useful in noisy OCR settings
- Low means:
  - the predicted value is far from the target
- Best use:
  - most relevant for `vendor_name` and `address`
  - less informative than exact match for numeric fields

## Per-Row Detail Metrics

These appear in `*_details.csv`.

### `predicted`

- What it is: the pipeline output for a field
- Why it matters: supports qualitative debugging and example-based analysis
- High or low meaning: not itself a score

### `ground_truth`

- What it is: the label from the dataset
- Why it matters: baseline for comparison
- High or low meaning: not itself a score

### `exact_match`

- What it is: whether the normalized prediction exactly equals normalized ground truth
- Why it matters: easy-to-read correctness signal
- `TRUE` means:
  - the field is correct after normalization
- `FALSE` means:
  - the field is incorrect, incomplete, malformed, or missing

### `fuzzy_score`

- What it is: similarity between the predicted and true value
- Why it matters: shows whether a wrong answer is still close
- High means:
  - prediction is close to ground truth
- Low means:
  - prediction is far from ground truth
- Practical reading:
  - `90-100`: very close
  - `60-89`: partially correct
  - below `60`: weak match

### `extracted`

- What it is: whether the system returned a non-null value
- Why it matters: distinguishes a missed field from a wrong field
- `TRUE` means:
  - the system attempted extraction
- `FALSE` means:
  - the field was left empty or null

### `image_path`

- What it is: the receipt image associated with the row
- Why it matters: supports targeted error analysis and presentation examples

### `mode`

- What it is: which of the four pipeline configurations produced the row
- Why it matters: allows direct system comparison

## Combined Overall Summary Metrics

These appear in `combined_mode_overall_summary.csv`.

### `mode`

- What it is: one of the four evaluation configurations
- Why it matters: comparison axis for the full experiment

### `num_records_evaluated`

- What it is: number of receipts successfully processed in that mode
- Why it matters:
  - indicates stability and sample size
- High means:
  - more stable execution
  - more trustworthy aggregate results
- Low means:
  - more failures or skipped records
  - lower confidence in the summary

### `num_field_rows`

- What it is: total number of field comparisons evaluated
- Why it matters:
  - shows the volume of evidence behind a summary
- High means:
  - more field-level evidence
- Low means:
  - fewer successful comparisons were included

### `num_entities_extracted`

- What it is: total number of non-null predictions across all fields and records
- Why it matters:
  - overall extraction coverage across the dataset
- High means:
  - more information is being extracted
- Low means:
  - many fields are being missed
- Caveat:
  - should be interpreted together with accuracy

### `overall_exact_match_accuracy`

- What it is: mean exact-match rate across all evaluated field rows
- Why it matters:
  - one headline correctness number for the mode
- High means:
  - more fields are exactly correct overall
- Low means:
  - many fields are incorrect or missing
- Caveat:
  - hides field-level differences, so it should be paired with per-field summaries

### `overall_avg_fuzzy_score`

- What it is: average fuzzy score across all evaluated field rows
- Why it matters:
  - useful as a softer global quality measure
- High means:
  - outputs are generally close to the target
- Low means:
  - outputs are broadly weak or noisy
- Caveat:
  - can be less meaningful than exact match for numerical fields

## Pivot Tables

These are presentation-friendly versions of the same metrics.

### `combined_exact_match_pivot.csv`

- Why it is relevant:
  - easiest side-by-side comparison of exact accuracy per field across all modes
- High value means:
  - that mode performs well on that field
- Low value means:
  - that mode struggles on that field

### `combined_fuzzy_score_pivot.csv`

- Why it is relevant:
  - easiest comparison of text-heavy field quality across modes
- High value means:
  - outputs are close to the target text
- Low value means:
  - outputs are far from the target text

### `combined_extracted_count_pivot.csv`

- Why it is relevant:
  - easiest comparison of extraction coverage across modes
- High value means:
  - the field is extracted often in that mode
- Low value means:
  - the field is often missed

## Error Files

These appear as `*_errors.csv`.

### `error`

- What it is: runtime failure reason for a specific image and mode
- Why it matters:
  - measures operational robustness, not extraction quality
- High number of error rows means:
  - that mode is unstable or environment-sensitive
- Low number of error rows means:
  - that mode is more reliable

## How To Interpret Results Properly

The most useful combinations are:

- High extraction count + high exact match:
  - strong and reliable performance
- High extraction count + low exact match:
  - aggressive but noisy extraction
- Low extraction count + high exact match:
  - conservative but precise extraction
- Low extraction count + low exact match:
  - weak field performance
- High fuzzy score + low exact match:
  - often close, but not normalized or not fully correct

## How This Maps To The Challenge

The challenge explicitly asks for:

- number of entities extracted
- correctness of extracted entities

This is directly addressed by:

- extraction count:
  - `num_entities_extracted`
- correctness:
  - `exact_match_accuracy`

The fuzzy score adds useful nuance for text-heavy fields such as vendor names and addresses.
