# Tests

A small pytest suite for `business_rules.py` - the age/income/spending
thresholds and marketing logic shared by the analysis script and the
dashboard. See AUDIT_REPORT.md (section 9, remediation log, item 11)
for why this exists.

## Running

```bash
pip install pytest
pytest tests/
```

## What's covered

- Every boundary of `categorize_age`, `categorize_income`, and
  `categorize_spending` (the exact values where a customer flips from
  one bucket to the next)
- Each branch of `segment_priority` and `generate_marketing_strategy`
- `generate_product_suggestions`, including the one number in that
  function that's actually computed from data (`spending_index`), and
  the zero-population edge case

## What's not covered

This suite only tests `business_rules.py`. The three main scripts
(`generator.py`, `customer_segmentation_analysis.py`,
`customer_segmentation_dashboard.py`) aren't unit tested - they were
instead verified by actually running them end to end (see the
remediation log for the exact runs and numbers). Worth adding later if
this project keeps growing:

- `DataPreprocessor` (outlier handling, scaling)
- The DBSCAN parameter grid search logic
- A smoke test that runs the full pipeline on a tiny synthetic dataset
  and checks it produces 5 clusters
