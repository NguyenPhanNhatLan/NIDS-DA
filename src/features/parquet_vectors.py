"""Validate and unpack one Arrow batch from legacy or Spark list Parquet."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from features.common_features import COMMON_FEATURES


def vector_matrix(vectors, context='', allow_nonfinite=False):
    dimension = len(COMMON_FEATURES)
    kind = vectors.type
    if vectors.null_count:
        raise ValueError(f'Null feature vectors in {context}')
    if pa.types.is_fixed_size_list(kind):
        if kind.list_size != dimension:
            raise ValueError(f'Expected {dimension} features in {context}')
    elif pa.types.is_list(kind) or pa.types.is_large_list(kind):
        lengths = pc.list_value_length(vectors).to_numpy(zero_copy_only=False)
        if not np.all(lengths == dimension):
            raise ValueError(f'Expected {dimension} features in {context}')
    else:
        raise ValueError(f'Expected list feature schema in {context}')

    flat = vectors.flatten()
    if flat.null_count:
        raise ValueError(f'Null feature values in {context}')
    matrix = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(vectors), dimension)
    if not allow_nonfinite and not np.isfinite(matrix).all():
        raise ValueError(f'NaN/Inf features in {context}')
    return matrix
