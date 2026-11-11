import unittest
import numpy as np
from houseprice import demo_data, evaluate, validate

class HousingTests(unittest.TestCase):
    def test_holdout_and_baseline(self):
        report = evaluate(demo_data())
        self.assertEqual(report["test_rows"], 100)
        self.assertLess(report["models"]["linear_regression"]["mae_usd"],
                        report["models"]["mean_baseline"]["mae_usd"])
        for metrics in report["models"].values():
            self.assertTrue(all(np.isfinite(list(metrics.values()))))

    def test_reproducible(self):
        self.assertEqual(evaluate(demo_data()), evaluate(demo_data()))

    def test_invalid_data(self):
        original = demo_data()
        cases = [original.iloc[:5], original.drop(columns="size_sqft")]
        for column, value in [("price_usd", np.nan), ("age_years", -1),
                              ("size_sqft", 0), ("bedrooms", 1.5), ("price_usd", np.inf)]:
            bad = original.copy()
            bad[column] = bad[column].astype(float)
            bad.loc[0, column] = value
            cases.append(bad)
        constant = original.copy()
        constant["price_usd"] = 1
        cases.append(constant)
        import pandas as pd
        cases.append(pd.concat([original, original.iloc[:1]]))
        for data in cases:
            with self.subTest():
                with self.assertRaises(ValueError):
                    validate(data)

if __name__ == "__main__":
    unittest.main()
