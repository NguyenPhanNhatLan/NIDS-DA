"""Source-fitted preprocessing for the 10 proposal features."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def signed_log1p(features):
    # Window sentinels can be negative, so preserve their sign.
    values = np.asarray(features, dtype=np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def proposal_processor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("log", FunctionTransformer(signed_log1p)),
        ("scaler", RobustScaler()),
    ])
