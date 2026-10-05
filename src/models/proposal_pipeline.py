"""Source-fitted preprocessing for the 5 proposal features."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def signed_log1p(features):
    # Window sentinels can be negative, so preserve their sign.
    values = np.asarray(features, dtype=np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def proposal_processor():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log", FunctionTransformer(signed_log1p)),
            ("scaler", RobustScaler()),
        ]
    )


# Only O(feature_count) parameters are retained from distributed Spark fitting.
# NumPy here transforms a caller-supplied batch; it never loads a dataset.
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedSourceProcessor:
    medians: tuple[float, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    relative_error: float
    spark_version: str
    common_feature_config_sha256: str
    data_revision: str = "spark_data_v1"

    def transform(self, features):
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.medians):
            raise ValueError("Feature schema mismatch")
        values = np.where(np.isfinite(values), values, np.nan)
        values = np.where(np.isnan(values), np.asarray(self.medians), values)
        logged = np.sign(values) * np.log1p(np.abs(values))
        return ((logged - np.asarray(self.centers)) / np.asarray(self.scales)).astype(
            np.float32
        )
