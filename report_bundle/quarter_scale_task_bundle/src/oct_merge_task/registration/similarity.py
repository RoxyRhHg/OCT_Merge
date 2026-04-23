from __future__ import annotations

import numpy as np


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_view = np.asarray(a, dtype=np.float32)
    b_view = np.asarray(b, dtype=np.float32)
    a_centered = a_view - float(a_view.mean())
    b_centered = b_view - float(b_view.mean())
    denom = float(np.sqrt((a_centered * a_centered).sum() * (b_centered * b_centered).sum()))
    if denom == 0.0:
        return float("-inf")
    return float((a_centered * b_centered).sum() / denom)
