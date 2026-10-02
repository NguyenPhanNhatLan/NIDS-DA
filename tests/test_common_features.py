import unittest

import numpy as np
import pandas as pd

from features.common_features import (
    COMMON_FEATURES,
    fit_common_feature_pipeline,
    load_common_feature_config,
    transform_common_features,
)


class CommonFeatureTests(unittest.TestCase):
    def test_common_order_dimensions_and_train_only_fit(self):
        config = load_common_feature_config()
        frames = {}
        for domain in ("unsw", "cicids"):
            frames[domain] = pd.DataFrame({item[domain]: [1.0, np.inf, 3.0] for item in config})
            frames[domain]["label"] = [0, 1, 0]
        outputs = {}
        for domain, train in frames.items():
            state = fit_common_feature_pipeline(train, domain)
            validation = train.copy()
            validation.loc[0, config[0][domain]] = np.inf
            first = transform_common_features(validation, domain, state)
            second = transform_common_features(validation, domain, state)
            self.assertTrue(first.equals(second))
            self.assertEqual(state, fit_common_feature_pipeline(train, domain))
            self.assertEqual(list(first.columns), [*COMMON_FEATURES, "label"])
            self.assertTrue(np.isfinite(first[list(COMMON_FEATURES)].to_numpy()).all())
            outputs[domain] = first
        self.assertEqual(outputs["unsw"].shape[1], outputs["cicids"].shape[1])

    def test_missing_column_and_no_implicit_fitting(self):
        config = load_common_feature_config()
        frame = pd.DataFrame({item["unsw"]: [1] for item in config})
        frame["label"] = [0]
        with self.assertRaisesRegex(ValueError, "training-fitted"):
            transform_common_features(frame, "unsw")
        with self.assertRaisesRegex(ValueError, "Missing configured"):
            fit_common_feature_pipeline(frame.drop(columns=config[0]["unsw"]), "unsw")


if __name__ == "__main__":
    unittest.main()
