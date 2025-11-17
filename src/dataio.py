import os
import h5py
import numpy as np
from typing import Dict, Tuple, Optional


class EEH5Dataset:
    """
    Loader for parent dset4.h5/dset7.h5 layout.

    HDF5 structure (per split):
      training/
        input   : (Ntr, F)  float32  -> features (log-domain for many entries)
        xopt    : (Ntr, U)  float32  -> labels (log-power)
        objval  : (Ntr,)    float32  -> optimal objective value (optional)
      validation/
        input, xopt, objval
      test/
        input, xopt, objval
        SCA     : (Nte,)    float32  -> first-order baseline objective (optional)
        SCAmax  : (Nte,)    float32  -> baseline variant (optional)

    Usage:
      ds = EEH5Dataset("data/dset4.h5")
      Xtr, Ytr, meta_tr = ds.get_split("train")
      Xva, Yva, meta_va = ds.get_split("val")
      Xte, Yte, meta_te = ds.get_split("test")

    Notes:
    - Labels Y are already in LOG-POWER domain in dset4.h5. Do NOT log them again.
    - Many feature columns are log-domain as well (e.g., log-gains).
    """

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 dataset not found: {path}")
        self._inspect()

    def _inspect(self) -> None:
        with h5py.File(self.path, "r") as f:
            self.has_train = "training" in f
            self.has_val   = "validation" in f
            self.has_test  = "test" in f
            self.info: Dict[str, Dict[str, Tuple[int, ...]]] = {}
            for split in ("training", "validation", "test"):
                if split in f:
                    g = f[split]
                    self.info[split] = {k: tuple(g[k].shape) for k in g.keys()}

    def get_split(self, which: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        key = {"train": "training", "val": "validation", "test": "test"}[which]
        with h5py.File(self.path, "r") as f:
            if key not in f:
                raise KeyError(f"Split '{which}' not found in HDF5 (expected group '{key}')")
            g = f[key]
            X = np.array(g["input"], dtype=np.float32)   # (N, F)
            Y = np.array(g["xopt"],  dtype=np.float32)   # (N, U)  (LOG-POWER)
            meta: Dict[str, np.ndarray] = {}
            for k in ("objval", "SCA", "SCAmax"):
                if k in g:
                    meta[k] = np.array(g[k], dtype=np.float32)
        return X, Y, meta

    # ---------- Optional helpers (for consistent preprocessing) ----------

    @staticmethod
    def compute_scaler(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-feature mean/std on TRAIN features only.
        Returns (mu, sigma) with shape (1, F). sigma is never < 1e-8.
        """
        mu = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)
        return mu.astype(np.float32), sigma.astype(np.float32)

    @staticmethod
    def apply_scaler(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Apply standardization: (X - mu) / sigma. Shapes must broadcast (N, F) vs (1, F).
        """
        return (X - mu) / sigma

    @staticmethod
    def save_scaler(dirpath: str, mu: np.ndarray, sigma: np.ndarray) -> None:
        os.makedirs(dirpath, exist_ok=True)
        np.save(os.path.join(dirpath, "mu.npy"), mu)
        np.save(os.path.join(dirpath, "sigma.npy"), sigma)

    @staticmethod
    def load_scaler(dirpath: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        mu_path = os.path.join(dirpath, "mu.npy")
        sg_path = os.path.join(dirpath, "sigma.npy")
        if os.path.exists(mu_path) and os.path.exists(sg_path):
            mu = np.load(mu_path).astype(np.float32)
            sigma = np.load(sg_path).astype(np.float32)
            return mu, sigma
        return None

    # ---------- (Rarely needed here, but kept for completeness) ----------

    @staticmethod
    def to_log(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Convert linear -> log. (Do NOT use this on labels from dset4.h5; they are already log.)"""
        return np.log(np.maximum(x, eps))

    @staticmethod
    def from_log(xlog: np.ndarray) -> np.ndarray:
        """Convert log -> linear."""
        return np.exp(xlog)
