"""Seed-level descriptive statistics and Student-t intervals."""

import numpy as np
from scipy.stats import t as student_t


def seed_statistics(values):
    """Two-sided t interval for the mean across independent run seeds."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite seed observations")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half_width = (
        float(student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
        if len(values) > 1
        else None
    )
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95": (
            [mean - half_width, mean + half_width] if half_width is not None else None
        ),
        "ci_method": "student_t_across_seeds",
    }

