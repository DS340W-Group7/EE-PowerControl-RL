import numpy as np

def rel_gap(y_pred: np.ndarray, y_opt: np.ndarray, eps: float = 1e-9) -> float:
    """
    Relative L2 gap (%) to the optimal target.

    Both y_pred and y_opt must be in the SAME DOMAIN:
    - either both in linear power domain
    - or both in log-power domain
    (eval.py now ensures they are compared in linear domain)

    Computes:
        gap = ||y_pred - y_opt|| / (||y_opt|| + eps) * 100%

    Also guards against NaN/inf values.
    """

    # convert to float64 for numerical safety
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_opt  = np.asarray(y_opt, dtype=np.float64)

    # replace NaN/inf with zeros
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    y_opt  = np.nan_to_num(y_opt,  nan=0.0, posinf=0.0, neginf=0.0)

    num = np.linalg.norm(y_pred - y_opt)
    den = np.linalg.norm(y_opt) + eps

    return float(100.0 * num / den)
