import torch
import torch.nn.functional as F
import numpy as np

# -----------------------------
# Wireless environment utilities
# -----------------------------
# Assumptions for dset4.h5:
# - First U*U features (by default starting at start=0) are *log*-gains of G (row-major).
# - Labels xopt are *log*-powers.
# - We exponentiate logs -> linear before computing SINR/rates.
# - Reward = WSEE = sum_i R_i / (p_i + p_c), with R_i = log2(1 + SINR_i).

# Numeric guardrails
_LOG_MIN, _LOG_MAX = -40.0, 40.0    # for exp safety
_LIN_MIN, _LIN_MAX = 0.0, 1e6       # linear clamp
_DEN_MIN, _DEN_MAX = 1e-9, 1e9
_RATE_MAX = 60.0                    # cap on log2(1+SINR)


def unpack_G(Xb: torch.Tensor, users: int, start: int = 0) -> torch.Tensor:
    """
    Extract a (U,U) *log*-gain block from features and return *linear* gains.

    Args:
        Xb    : (B, F) float32 features
        users : U (number of users)
        start : starting column index of the U*U G block (default 0)

    Returns:
        G : (B, U, U) linear power gains (>=0)
    """
    B, F = Xb.shape
    need = users * users
    if start + need > F:
        raise ValueError(f"Need {need} cols from start={start}, but F={F}.")
    Glog = Xb[:, start:start + need].contiguous().view(B, users, users)  # log-gains

    # Clamp to avoid overflow, then exp -> linear gains
    G = torch.exp(torch.clamp(torch.nan_to_num(Glog, nan=0.0), min=_LOG_MIN, max=_LOG_MAX))
    G = torch.clamp(G, min=_LIN_MIN, max=_LIN_MAX)
    return G


def rates_log2(G: torch.Tensor, p: torch.Tensor, sigma2: float = 1.0) -> torch.Tensor:
    """
    Compute per-user rates R_i = log2(1 + SINR_i).

    Args:
        G      : (B, U, U) linear gains
        p      : (B, U)    linear transmit powers (>=0)
        sigma2 : noise power (linear)

    Returns:
        Ri : (B, U) rates in bit/s/Hz
    """
    B, U, _ = G.shape
    p = torch.clamp(torch.nan_to_num(p, nan=0.0, posinf=_LIN_MAX, neginf=0.0), min=_LIN_MIN, max=_LIN_MAX)

    # Received power from each TX at each RX: multiply along TX dimension (last axis)
    # rx_power[b, i, j] = G[b, i, j] * p[b, j]
    rx_power = G * p.view(B, 1, U)
    rx_power = torch.nan_to_num(rx_power, nan=0.0, posinf=_LIN_MAX, neginf=0.0)

    signal = torch.diagonal(rx_power, dim1=1, dim2=2)     # (B, U) -> G_ii * p_i
    total  = rx_power.sum(dim=2)                           # (B, U)
    interf = total - signal

    denom = torch.clamp(torch.nan_to_num(interf + float(sigma2),
                                         nan=1.0, posinf=_DEN_MAX, neginf=1.0), min=_DEN_MIN, max=_DEN_MAX)
    num   = torch.clamp(torch.nan_to_num(signal, nan=0.0, posinf=_DEN_MAX, neginf=0.0), min=_LIN_MIN, max=_DEN_MAX)

    sinr = torch.clamp(torch.nan_to_num(num / denom, nan=0.0, posinf=_LIN_MAX, neginf=0.0), min=_LIN_MIN, max=_LIN_MAX)
    Ri = torch.log1p(sinr) / np.log(2.0)                   # log2(1 + SINR)
    Ri = torch.clamp(torch.nan_to_num(Ri, nan=0.0, posinf=_RATE_MAX, neginf=0.0), min=0.0, max=_RATE_MAX)
    return Ri


def wsee_reward(Ri: torch.Tensor, p: torch.Tensor, p_c: float = 1e-3) -> torch.Tensor:
    """
    Weighted-sum energy efficiency: sum_i R_i / (p_i + p_c).

    Args:
        Ri   : (B, U) rates
        p    : (B, U) linear powers
        p_c  : small circuit power to stabilize denominator

    Returns:
        r : (B,) scalar reward per sample
    """
    denom = torch.clamp(torch.nan_to_num(p + p_c, nan=1.0, posinf=_DEN_MAX, neginf=1.0), min=_DEN_MIN, max=_DEN_MAX)
    r = torch.sum(Ri / denom, dim=1)
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r


def powers_from_policy(logp: torch.Tensor, pmax_log: torch.Tensor) -> torch.Tensor:
    """
    Convert network outputs (log-power) to valid linear powers, clipped by exp(pmax_log).

    Args:
        logp     : (B, U) network outputs in *log-power* domain
        pmax_log : (B,) or (B,1) per-sample *log-power* cap (derived from labels)

    Returns:
        p : (B, U) linear powers in [0, exp(pmax_log)]
    """
    logp = torch.clamp(torch.nan_to_num(logp, nan=0.0), min=_LOG_MIN, max=_LOG_MAX)
    p = torch.exp(logp)  # linear

    if pmax_log.ndim == 1:
        pmax_log = pmax_log.view(-1, 1)
    pmax_log = torch.clamp(torch.nan_to_num(pmax_log, nan=0.0), min=_LOG_MIN, max=_LOG_MAX)
    pmax = torch.exp(pmax_log)

    p = torch.minimum(p, pmax.expand_as(p))
    p = torch.clamp(p, min=_LIN_MIN, max=_LIN_MAX)
    return p
