import unittest

import numpy as np
import pandas as pd

from features.common_features import (
    COMMON_FEATURES,
    canonicalize_frame,
    load_common_feature_config,
)


class CommonFeatureTests(unittest.TestCase):
    def test_order_and_missing_values_remain_unfitted(self):
        config = load_common_feature_config()
        self.assertEqual(len(COMMON_FEATURES), 5)
        self.assertEqual([item["canonical_name"] for item in config], list(COMMON_FEATURES))
        for domain in ("unsw", "cicids"):
            frame = pd.DataFrame({item[domain]: [1.0, np.inf, 3.0] for item in config})
            frame["label"] = [0, 1, 0]
            output = canonicalize_frame(frame, domain, config)
            self.assertEqual(list(output.columns), [*COMMON_FEATURES, "label"])
            self.assertEqual(output.shape, (3, len(COMMON_FEATURES) + 1))
            self.assertTrue(output[list(COMMON_FEATURES)].iloc[1].isna().all())
            self.assertTrue(output.equals(canonicalize_frame(frame, domain, config)))

    def test_missing_source_column_fails(self):
        config = load_common_feature_config()
        frame = pd.DataFrame({item["unsw"]: [1.0] for item in config})
        frame["label"] = [0]
        with self.assertRaisesRegex(ValueError, "Missing unsw column"):
            canonicalize_frame(frame.drop(columns=config[0]["unsw"]), "unsw", config)


if __name__ == "__main__":
    unittest.main()
