"""House price prediction on the California Housing dataset.

Compares three regression models with 5-fold cross-validation on a training
split, evaluates the best one on a held-out test split, and saves the metrics
and a plot to the ``results/`` folder.

The dataset is downloaded automatically by scikit-learn the first time the
script runs (internet access is needed once) and is then cached on disk.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, so no display is needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DOLLARS_PER_UNIT = 100_000  # the dataset expresses prices in units of $100,000
RESULTS_DIR = Path("results")


def load_data():
    """Return the features (DataFrame) and the target (median house value)."""
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target


def build_models():
    """Return the models to compare, keyed by name."""
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
    }


def compare_models(models, X_train, y_train, folds=CV_FOLDS):
    """Cross-validate every model on the training data and return a summary table."""
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=splitter,
            scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"),
        )
        rows.append(
            {
                "model": name,
                "cv_rmse_usd": -scores["test_neg_root_mean_squared_error"].mean()
                * DOLLARS_PER_UNIT,
                "cv_mae_usd": -scores["test_neg_mean_absolute_error"].mean()
                * DOLLARS_PER_UNIT,
                "cv_r2": scores["test_r2"].mean(),
            }
        )
    return pd.DataFrame(rows)


def evaluate_on_test(model, X_train, y_train, X_test, y_test):
    """Fit on the training data and return the test metrics and the predictions."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "rmse_usd": mean_squared_error(y_test, y_pred) ** 0.5 * DOLLARS_PER_UNIT,
        "mae_usd": mean_absolute_error(y_test, y_pred) * DOLLARS_PER_UNIT,
        "r2": r2_score(y_test, y_pred),
    }
    return metrics, y_pred


def plot_actual_vs_predicted(y_test, y_pred, title, path):
    """Save a scatter plot of actual against predicted prices."""
    actual = y_test * DOLLARS_PER_UNIT
    predicted = y_pred * DOLLARS_PER_UNIT
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, s=4, alpha=0.3)
    ax.plot([low, high], [low, high], color="red", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual median house value (USD)")
    ax.set_ylabel("Predicted median house value (USD)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Training rows: {len(X_train)}, test rows: {len(X_test)}")

    models = build_models()
    comparison = compare_models(models, X_train, y_train)
    print(f"\n--- {CV_FOLDS}-fold cross-validation on the training data ---")
    print(comparison.round(3).to_string(index=False))

    best_name = comparison.loc[comparison["cv_rmse_usd"].idxmin(), "model"]
    metrics, y_pred = evaluate_on_test(models[best_name], X_train, y_train, X_test, y_test)
    print(f"\n--- Best model by CV RMSE: {best_name} (evaluated once on the test set) ---")
    for key, value in metrics.items():
        print(f"{key}: {value:,.3f}")

    comparison.to_csv(RESULTS_DIR / "cv_comparison.csv", index=False)
    pd.DataFrame([{"model": best_name, **metrics}]).to_csv(
        RESULTS_DIR / "test_metrics.csv", index=False
    )
    plot_actual_vs_predicted(
        y_test, y_pred, f"{best_name}: actual vs predicted", RESULTS_DIR / "actual_vs_predicted.png"
    )
    print(f"\nSaved results to the '{RESULTS_DIR}' folder.")


if __name__ == "__main__":
    main()
