"""Fixed common preprocessing for the proposal's controlled comparison."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def signed_log1p(features):
    # Mapped TCP-window sentinels may be negative; preserve sign explicitly.
    values=np.asarray(features,dtype=np.float64)
    return np.sign(values)*np.log1p(np.abs(values))


def proposal_processor():
    return Pipeline([
        ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('log',FunctionTransformer(signed_log1p)),
        ('scaler',RobustScaler()),
    ])
