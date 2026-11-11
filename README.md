# House Price Prediction
A small, reproducible Python regression project with input validation, a held-out test set and a mean-price baseline.

**Status:** educational prototype. The default 500-row dataset is synthetic, generated locally. Results are not evidence of real housing-market accuracy.

## Quick start
Requires Python 3.10 or later.

```bash
git clone https://github.com/muhammadaliraza126/House-Price-Prediction.git
cd House-Price-Prediction
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python houseprice.py
python houseprice.py --output reports/demo.json
python -m unittest discover -s tests -v
```

No API key, paid service or dataset download is required.

## Use your own CSV
```bash
python houseprice.py --csv data/houses.csv --output reports/houses.json
```

| Column | Meaning | Validation |
| --- | --- | --- |
| size_sqft | Floor area, square feet | Positive finite number |
| bedrooms | Bedroom count | Positive whole number |
| age_years | Property age, years | Nonnegative finite number |
| price_usd | Sale price in US dollars | Positive finite number |

Provide at least 20 complete, unique rows with varying prices. Missing columns, nonnumeric values, missing/infinite values, invalid ranges and duplicate rows are rejected. Extra columns are ignored.

## Evaluation
An 80/20 split uses a fixed seed (default 42). Both models are fitted only on training rows. The JSON report contains sample counts, data source, seed, MAE and RMSE in USD, and R² for linear regression and the training-mean baseline. Lower MAE/RMSE is better; R² can be negative.

Synthetic data follows a mostly linear formula, so good demo scores are expected. R² is not a percentage accuracy guarantee. Run the script to obtain actual results in your environment.

## Limitations
- No location features, market trends, inflation adjustment or uncertainty intervals.
- Random splitting is suitable for this synthetic demo; real transactions may require temporal or property-group splits to avoid leakage.
- Similar properties can leak information even when exact duplicate rows are removed.
- No XGBoost, dashboard, deployed API or production valuation claims.
- Dependencies have bounded compatible ranges; these are not a fully locked environment.

## Contribute
See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports should include a small synthetic example and the command that failed.

Next improvements: licensed real-world data, temporal validation, categorical location features and error analysis. These are planned, not implemented.

## Development transparency
This project is developed with AI assistance. Changes should be reviewed and tested; synthetic results should never be presented as real-world performance.

## License
MIT; see [LICENSE](LICENSE).
