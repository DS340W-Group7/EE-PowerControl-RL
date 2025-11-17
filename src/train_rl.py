# src/train_rl.py
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataio import EEH5Dataset
from models import MLP
from env import unpack_G, rates_log2, wsee_reward, powers_from_policy


# ----------------- helpers -----------------

def _maybe_load_scaler(scaler_dir: str | None):
    """Load mu/sigma saved by train_ann.py (if present)."""
    if not scaler_dir:
        return None, None
    mu_p = os.path.join(scaler_dir, "mu.npy")
    sg_p = os.path.join(scaler_dir, "sigma.npy")
    if os.path.exists(mu_p) and os.path.exists(sg_p):
        mu = np.load(mu_p).astype(np.float32)
        sigma = np.load(sg_p).astype(np.float32)
        sigma = np.where(sigma < 1e-8, 1e-8, sigma).astype(np.float32)
        return mu, sigma
    return None, None


def _apply_scaler(X: np.ndarray, mu: np.ndarray | None, sigma: np.ndarray | None) -> np.ndarray:
    if mu is None or sigma is None:
        return X
    return (X - mu) / sigma


def _first_batch_debug(xb, yopt, base, model, users, sigma2, p_c, g_start, delta_scale):
    """Quick sanity check on domains and reward before training loop."""
    with torch.no_grad():
        G = unpack_G(xb, users=users, start=g_start)
        pmax_log = torch.max(yopt, dim=1).values

        base_logp = base(xb)
        model_logp = model(xb)
        logp = base_logp + delta_scale * (model_logp - base_logp)

        p = powers_from_policy(logp, pmax_log)
        Ri = rates_log2(G, p, sigma2=sigma2)
        r = wsee_reward(Ri, p, p_c=p_c)

        print("[debug] G min/max:", float(G.min()), float(G.max()))
        print("[debug] p min/max:", float(p.min()), float(p.max()))
        print("[debug] Ri min/max:", float(Ri.min()), float(Ri.max()))
        print("[debug] r mean:", float(r.mean()))
        if (torch.isnan(G).any() or torch.isnan(p).any()
                or torch.isnan(Ri).any() or torch.isnan(r).any()):
            raise RuntimeError("NaN in first-batch debug — check scaling and g_start.")


# ----------------- training -----------------

def train_rl(
    data_path: str,
    init_ckpt: str,
    outdir: str,
    users: int = 4,
    steps: int = 8000,
    bs: int = 512,
    lr: float = 1e-5,
    sigma2: float = 1.0,
    p_c: float = 1e-3,
    seed: int = 0,
    g_start: int = 0,
    scaler_dir: str | None = None,
    delta_scale: float = 0.10,   # size of RL correction in log-domain
):
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- data -----
    ds = EEH5Dataset(data_path)
    Xtr, Ytr, _ = ds.get_split("train")  # Ytr are LOG-POWER labels

    # ----- scaling (use ANN scaler) -----
    if scaler_dir is None:
        scaler_dir = os.path.dirname(init_ckpt)  # default: same dir as ckpt
    mu, sigma = _maybe_load_scaler(scaler_dir)
    if mu is not None:
        Xtr = _apply_scaler(Xtr, mu, sigma)
        print(f"[rl] applied scaler from: {scaler_dir}")
    else:
        print("[rl] no scaler found; using raw features")

    # ----- loader -----
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=bs, shuffle=True, drop_last=True
    )

    # ----- models -----
    # trainable policy initialized from ANN
    model = MLP(in_dim=Xtr.shape[1], out_dim=Ytr.shape[1])
    model.load_state_dict(torch.load(init_ckpt, map_location="cpu"))
    model.train()

    # frozen ANN baseline (anchor for delta-logp)
    base = MLP(in_dim=Xtr.shape[1], out_dim=Ytr.shape[1])
    base.load_state_dict(torch.load(init_ckpt, map_location="cpu"))
    for p in base.parameters():
        p.requires_grad = False
    base.eval()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    clip_norm = 1.0

    # ----- one-time sanity check -----
    xb0, y0 = next(iter(train_loader))
    _first_batch_debug(xb0.float(), y0.float(), base, model, users, sigma2, p_c, g_start, delta_scale)

    ema_reward, ema_beta = None, 0.98
    step = 0

    while step < steps:
        for xb, yopt in train_loader:
            xb = xb.float()
            yopt = yopt.float()  # LOG-POWER labels

            # 1) channel gains (linear)
            G = unpack_G(xb, users=users, start=g_start)

            # 2) per-sample cap (log)
            pmax_log = torch.max(yopt, dim=1).values

            # 3) policy forward (delta-logp anchored to base ANN)
            with torch.no_grad():
                base_logp = base(xb)           # fixed ANN output
            model_logp = model(xb)             # trainable output
            logp = base_logp + delta_scale * (model_logp - base_logp)

            # 4) convert to linear powers
            p = powers_from_policy(logp, pmax_log)

            # 5) reward = WSEE
            Ri = rates_log2(G, p, sigma2=sigma2)
            r = wsee_reward(Ri, p, p_c=p_c)

            if torch.isnan(r).any():
                raise RuntimeError("NaN reward — check g_start/scaling/clamps.")

            # 6) optimize -reward
            loss = -torch.mean(r)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()

            rmean = float(r.mean().detach().cpu().numpy())
            ema_reward = rmean if ema_reward is None else (ema_beta * ema_reward + (1 - ema_beta) * rmean)

            step += 1
            if step % 100 == 0:
                print(f"[RL] step {step:6d} | loss {loss.item():.6f} | reward {rmean:.6f} | ema {ema_reward:.6f}")
            if step >= steps:
                break

    # save RL-enhanced policy
    ckpt_out = os.path.join(outdir, "model_rl.pt")
    torch.save(model.state_dict(), ckpt_out)
    print(f"[rl] saved: {ckpt_out}")


# ----------------- CLI -----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", required=True, help="Path to ANN checkpoint (e.g., results/ann2/model.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--pc", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--g_start", type=int, default=0, help="start column index of U*U log-G block")
    ap.add_argument("--scaler_dir", type=str, default=None,
                    help="dir containing mu.npy/sigma.npy (defaults to init ckpt dir)")
    ap.add_argument("--delta_scale", type=float, default=0.10,
                    help="size of RL correction blended with ANN baseline (0.05–0.2 typical)")
    args = ap.parse_args()

    train_rl(
        data_path=args.data,
        init_ckpt=args.init,
        outdir=args.out,
        users=args.users,
        steps=args.steps,
        bs=args.bs,
        lr=args.lr,
        sigma2=args.sigma2,
        p_c=args.pc,
        seed=args.seed,
        g_start=args.g_start,
        scaler_dir=args.scaler_dir,
        delta_scale=args.delta_scale,
    )
