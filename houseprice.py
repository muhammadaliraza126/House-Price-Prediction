"""Reproducible educational housing regression; synthetic demo, not a valuation service."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["size_sqft", "bedrooms", "age_years"]
TARGET = "price_usd"

def demo_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "size_sqft": rng.uniform(600, 4000, n),
        "bedrooms": rng.integers(1, 7, n),
        "age_years": rng.integers(0, 100, n),
    })
    data[TARGET] = (60000 + 180 * data.size_sqft + 12000 * data.bedrooms
                    - 700 * data.age_years + rng.normal(0, 25000, n)).clip(10000)
    return data

def validate(data):
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    clean = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="raise")
    if len(clean) < 20:
        raise ValueError("At least 20 rows are required for a train/test split.")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Missing or infinite values are not allowed.")
    if (clean[["size_sqft", "bedrooms", TARGET]] <= 0).any().any():
        raise ValueError("Size, bedrooms and price must be positive.")
    if (clean.age_years < 0).any():
        raise ValueError("Age must be nonnegative.")
    if not (clean.bedrooms % 1 == 0).all():
        raise ValueError("Bedrooms must be whole numbers.")
    if clean.duplicated().any():
        raise ValueError("Duplicate rows may leak between train/test sets; remove them first.")
    if clean[TARGET].nunique() < 2:
        raise ValueError("Prices must vary.")
    return clean

def evaluate(data, seed=42):
    clean = validate(data)
    x_train, x_test, y_train, y_test = train_test_split(
        clean[FEATURES], clean[TARGET], test_size=0.2, random_state=seed)
    if y_test.nunique() < 2:
        raise ValueError("Test prices must vary to calculate meaningful R2.")
    report = {"train_rows": len(x_train), "test_rows": len(x_test), "seed": seed, "models": {}}
    for name, model in [("mean_baseline", DummyRegressor()), ("linear_regression", LinearRegression())]:
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        report["models"][name] = {
            "mae_usd": float(mean_absolute_error(y_test, prediction)),
            "rmse_usd": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "r2": float(r2_score(y_test, prediction)),
        }
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV with size_sqft, bedrooms, age_years, price_usd")
    parser.add_argument("--output", type=Path, help="Save a JSON report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        data = pd.read_csv(args.csv) if args.csv else demo_data(seed=args.seed)
        report = evaluate(data, args.seed)
        report["data_source"] = "user_csv" if args.csv else "synthetic_demo"
        result = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result + "\n", encoding="utf-8")
        print(result)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    main()
