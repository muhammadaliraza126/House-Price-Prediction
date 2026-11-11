# CSV Data Quality Checker

A standalone Python 3.10+ command-line tool. Uses only the standard library; no installation, API key or network request is required. Located in this repository as a reusable companion to the housing example.

## Run from the repository root

```bash
python tools/csv_quality.py examples/houses_synthetic.csv --numeric size_sqft bedrooms age_years price_usd
python tools/csv_quality.py examples/houses_synthetic.csv --output reports/quality.json
python -m unittest discover -s tests -v
```

The second command checks structural quality and blanks/duplicates; numeric checks require `--numeric`. Output files must not already exist. Choose a new filename for each saved report.

## Checks and report

- Counts empty or whitespace-only cells per column.
- Counts exact duplicate records after the first occurrence.
- Counts records whose field count differs from the header.
- Optionally counts nonnumeric or nonfinite values (including NaN and infinity).
- Flags a header-only dataset as empty.
- Rejects missing, blank or repeated column names and unknown numeric columns.

Exit codes: **0** means the selected checks passed, **1** means data-quality issues were found, **2** means an input/output or parsing error. JSON is printed for codes 0 and 1. Errors go to stderr.

Malformed-width rows are excluded from cell and duplicate counts. Blank numeric cells count only as blanks. Whitespace is considered when detecting blanks, but duplicates compare original field strings exactly. CSV records can span physical lines. UTF-8 with or without a BOM is supported. Header names are case-sensitive.

## Limits and privacy

No data leaves the machine. Reports contain column names and counts, not row values. Review column names before sharing reports. Input is never modified. The tool stores unique records in memory, so it is intended for small-to-medium CSV files, not unbounded datasets. It expects comma-separated UTF-8 CSV and does not infer schemas, business rules, date formats, value ranges or statistical outliers. A passing result does not prove a dataset is correct or suitable for machine learning.

## Example with known issues

```bash
python tools/csv_quality.py examples/quality_issues.csv --numeric amount
```

Expected: 5 records, 1 duplicate, 1 malformed-width record, 1 blank name and 2 invalid amounts. This synthetic fixture intentionally exits with code 1.

MIT license: see the repository LICENSE. Developed with AI assistance and covered by automated tests.
