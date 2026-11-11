"""Tests for the model comparison code, using a small synthetic dataset.

The real California Housing data is not used here, so the tests run quickly
and without internet access.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import houseprice


def make_synthetic_data(rows=300):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(rows, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - X["b"] + 0.1 * rng.normal(size=rows)
    return X, y


def test_build_models_names():
    assert list(houseprice.build_models()) == [
        "Linear Regression",
        "Random Forest",
        "Hist Gradient Boosting",
    ]


def test_compare_models_returns_one_row_per_model():
    X, y = make_synthetic_data()
    models = {"Linear": LinearRegression()}
    table = houseprice.compare_models(models, X, y, folds=3)

    assert list(table["model"]) == ["Linear"]
    assert {"cv_rmse_usd", "cv_mae_usd", "cv_r2"} <= set(table.columns)
    assert table.loc[0, "cv_r2"] > 0.9


def test_evaluate_on_test_returns_metrics_and_predictions():
    X, y = make_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    metrics, y_pred = houseprice.evaluate_on_test(
        LinearRegression(), X_train, y_train, X_test, y_test
    )

    assert set(metrics) == {"rmse_usd", "mae_usd", "r2"}
    assert len(y_pred) == len(y_test)
    assert metrics["r2"] > 0.9
