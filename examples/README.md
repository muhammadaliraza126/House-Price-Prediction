# CSV walkthrough

Run from the repository root:

```bash
python houseprice.py --csv examples/houses_synthetic.csv --output reports/example.json
```

The same JSON is printed and saved. `train_rows` should be 80 and `test_rows` 20. Compare `linear_regression` with `mean_baseline` on the same held-out rows.

- MAE: average absolute prediction error, in US dollars.
- RMSE: error in US dollars with a larger penalty for large mistakes.
- R2: comparison against variation in held-out prices; it can be negative and is not percentage accuracy.

This is synthetic demonstration data, not real sales or a market benchmark. The CSV was generated from this repository's `demo_data(n=100, seed=42)` and rounded to two decimals. No external dataset was used.

## Troubleshooting

| Error | What to check |
| --- | --- |
| Missing columns | Use size_sqft, bedrooms, age_years, price_usd exactly |
| At least 20 rows | Include at least 20 complete records |
| Missing or infinite values | Correct blank, NaN or infinite entries |
| Duplicate rows | Review and remove identical records |
| Unable to parse string | Remove currency symbols and text from numeric fields |
| File not found | Run from the repository root and check the CSV path |

Invalid input exits with a nonzero status and an error on stderr. Do not fill missing prices with invented values just to make validation pass.

## Run checks

```bash
python -m unittest discover -s tests -v
```
