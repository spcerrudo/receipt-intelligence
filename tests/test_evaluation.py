from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.entity_schema import ReceiptEntities
from src.evaluation import Evaluator


def test_evaluator_normalizes_exact_matches():
    evaluator = Evaluator()
    predicted = ReceiptEntities(
        vendor_name="ACME STORE",
        date="01/02/2024",
        total="$12.5",
        address="123 Market Street",
    )
    truth = ReceiptEntities(
        vendor_name="Acme Store",
        date="2024-02-01",
        total="12.50",
        address="123  Market   Street",
    )

    rows = evaluator.compare(predicted, truth)
    row_map = {row["field"]: row for row in rows}

    assert row_map["vendor_name"]["fuzzy_score"] == 100.0
    assert row_map["date"]["exact_match"] is True
    assert row_map["total"]["exact_match"] is True
