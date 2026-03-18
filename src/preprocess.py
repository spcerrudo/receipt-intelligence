from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray,
    apply_threshold: bool = True,
    resize_max_width: int = 1600,
) -> np.ndarray:
    """Apply lightweight preprocessing suitable for receipt OCR."""

    if image is None:
        raise ValueError("Input image is None")

    processed = image.copy()
    if processed.ndim == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    height, width = processed.shape[:2]
    if width > resize_max_width:
        scale = resize_max_width / float(width)
        processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    processed = cv2.GaussianBlur(processed, (3, 3), 0)

    if apply_threshold:
        processed = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
    return processed
