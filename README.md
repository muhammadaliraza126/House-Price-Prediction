# House Price Prediction

A small, reproducible machine learning project that predicts median house values on the California Housing dataset and compares three regression models.

## What it does

1. Loads the California Housing dataset through scikit-learn.
2. Splits the data into 80% training and 20% test sets.
3. Compares three models with 5-fold cross-validation on the training set only:
   - Linear Regression (with feature scaling)
   - Random Forest
   - Histogram Gradient Boosting
4. Picks the model with the lowest cross-validated RMSE, fits it on the training set, and evaluates it once on the held-out test set.
5. Saves the metrics and an actual-versus-predicted plot to the `results/` folder.

## Dataset

California Housing, derived from the 1990 U.S. census. Each row is a census block group (20,640 rows, 8 features): median income, house age, average rooms, average bedrooms, population, average occupancy, latitude and longitude. The target is the median house value, which the dataset states in units of $100,000; the script converts it to US dollars.

scikit-learn downloads the dataset automatically the first time the script runs, so an internet connection is needed once. It is then cached on your machine.

## Installation

```bash
git clone https://github.com/muhammadaliraza126/House-Price-Prediction.git
cd House-Price-Prediction
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python houseprice.py
```

The script prints the cross-validation table and the test metrics, and writes three files to `results/`:

| File | Contents |
| --- | --- |
| `cv_comparison.csv` | Cross-validated RMSE, MAE and R² for each model |
| `test_metrics.csv` | Test-set RMSE, MAE and R² for the best model |
| `actual_vs_predicted.png` | Scatter plot of actual against predicted prices |

## Results

This repository does not include pre-computed numbers. Run the script to see the results for your own machine; they are printed to the terminal and saved in `results/`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use a small synthetic dataset, so they run in seconds and need no internet connection.

## Limitations

- The data comes from the 1990 census, so predicted values do not reflect today's prices.
- The dataset caps prices at about $500,000, which limits how well any model can predict expensive areas.
- Latitude and longitude are used as plain numeric features, and no hyperparameter tuning has been done.
- This is a learning project, not a tool for pricing real properties.

## Project structure

```
houseprice.py          main script
tests/                 unit tests
requirements.txt       runtime dependencies
requirements-dev.txt   adds pytest for the tests
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
