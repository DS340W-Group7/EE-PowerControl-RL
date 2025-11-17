import argparse, os, numpy as np, torch
from dataio import EEH5Dataset
from models import MLP
from env import unpack_G, rates_log2, wsee_reward, powers_from_policy

# ---------- scaler helpers ----------
def _maybe_load_scaler(sdir):
    if not sdir: return None, None
    mp, sp = os.path.join(sdir,"mu.npy"), os.path.join(sdir,"sigma.npy")
    if os.path.exists(mp) and os.path.exists(sp):
        mu = np.load(mp).astype(np.float32)
        sg = np.load(sp).astype(np.float32)
        sg = np.where(sg < 1e-8, 1e-8, sg).astype(np.float32)
        return mu, sg
    return None, None

def _apply(X, mu, sg):
    return X if (mu is None or sg is None) else (X - mu)/sg

# ---------- prediction helper ----------
@torch.no_grad()
def _predict_logp(X, out_dim, ckpt, scaler_dir, base_ckpt=None, delta_scale=0.10):
    """
    Returns predicted LOG-powers, optionally blended with a base (ANN) checkpoint.
    """
    mu, sg = _maybe_load_scaler(scaler_dir if scaler_dir else os.path.dirname(ckpt))
    Xn = _apply(X, mu, sg)

    # main model
    model = MLP(in_dim=X.shape[1], out_dim=out_dim)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    Xt = torch.from_numpy(Xn).float()

    if base_ckpt:
        base = MLP(in_dim=X.shape[1], out_dim=out_dim)
        base.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
        base.eval()
        base_log = base(Xt).cpu().numpy()
        rl_log   = model(Xt).cpu().numpy()
        Yhat_log = (1.0 - delta_scale) * base_log + delta_scale * rl_log
    else:
        Yhat_log = model(Xt).cpu().numpy()

    # safety clamp to avoid overflow on exp
    Yhat_log = np.clip(Yhat_log, -20.0, 5.0)
    return Yhat_log

# ---------- objective evaluator ----------
def evaluate_objective(
    data_path, ann_ckpt, rl_ckpt, outdir,
    scaler_dir=None, rl_base=None, rl_delta=0.10,
    users=4, g_start=0, sigma2=1.0, p_c=1e-3, limit_test=0, bs=4096
):
    os.makedirs(outdir, exist_ok=True)

    # load test split
    ds = EEH5Dataset(data_path)
    Xte, Yte, meta = ds.get_split("test")    # Yte in LOG-power (BB)
    if limit_test and limit_test < Xte.shape[0]:
        Xte, Yte = Xte[:limit_test], Yte[:limit_test]
        for k in list(meta.keys()):
            meta[k] = meta[k][:limit_test]

    out_dim = Yte.shape[1]

    # helper to compute mean objective from a ckpt (blended if base provided)
    def obj_from_ckpt(ckpt, base_ckpt=None, delta_scale=0.10):
        # predict log-powers (optionally blended with base ANN)
        Yhat_log = _predict_logp(Xte, out_dim, ckpt, scaler_dir, base_ckpt, delta_scale)
        # per-sample cap in LOG domain (same as training)
        pmax_log = np.max(Yte, axis=1).astype(np.float32)  # (N,)

        N = Xte.shape[0]
        tot = 0.0
        for i in range(0, N, bs):
            xb = torch.from_numpy(Xte[i:i+bs]).float()
            logp = torch.from_numpy(Yhat_log[i:i+bs].astype(np.float32))
            pmax = torch.from_numpy(pmax_log[i:i+bs])

            G = unpack_G(xb, users=users, start=g_start)              # channels (linear)
            p = powers_from_policy(logp, pmax)                         # powers (linear)
            Ri = rates_log2(G, p, sigma2=sigma2)
            r  = wsee_reward(Ri, p, p_c=p_c)                           # (B,)
            tot += float(r.sum())
        return tot / float(N)

    # --- NEW: compute "env-optimal" objective by plugging the BB labels xopt into OUR env ---
    def obj_env_opt_from_labels():
        N = Xte.shape[0]
        tot = 0.0
        pmax_log = np.max(Yte, axis=1).astype(np.float32)  # cap stays consistent
        for i in range(0, N, bs):
            xb  = torch.from_numpy(Xte[i:i+bs]).float()
            logp_labels = torch.from_numpy(Yte[i:i+bs].astype(np.float32))  # BB log-powers
            pmax = torch.from_numpy(pmax_log[i:i+bs])

            G = unpack_G(xb, users=users, start=g_start)
            p = powers_from_policy(logp_labels, pmax)   # apply same cap pathway
            Ri = rates_log2(G, p, sigma2=sigma2)
            r  = wsee_reward(Ri, p, p_c=p_c)
            tot += float(r.sum())
        return tot / float(N)

    # compute objectives (all in OUR env)
    obj_env_opt = obj_env_opt_from_labels()
    obj_ann     = obj_from_ckpt(ann_ckpt, base_ckpt=None)
    obj_rl      = obj_from_ckpt(rl_ckpt, base_ckpt=rl_base, delta_scale=rl_delta)

    # relative gaps w.r.t. ENV-opt (lower is better; 0 means matched env-opt)
    def rel_gap_env(obj):
        return float(100.0 * max(obj_env_opt - obj, 0.0) / (abs(obj_env_opt) + 1e-9))

    gap_ann = rel_gap_env(obj_ann)
    gap_rl  = rel_gap_env(obj_rl)

    # also keep original H5 'objval' (may be different scale/assumptions)
    obj_h5_opt = float(np.mean(meta["objval"])) if "objval" in meta else float("nan")

    import pandas as pd
    df = pd.DataFrame([{
        # ENV metrics (the ones that matter for our pipeline)
        "env_obj_opt": obj_env_opt,
        "env_obj_ann": obj_ann,
        "env_obj_rl":  obj_rl,
        "env_rel_gap_ann_pct": gap_ann,
        "env_rel_gap_rl_pct":  gap_rl,

        # For transparency: the dataset's own 'objval' mean (not used for gap)
        "h5_objval_mean": obj_h5_opt,

        # bookkeeping
        "users": users, "sigma2": sigma2, "p_c": p_c,
        "rl_delta": rl_delta, "rl_base": rl_base if rl_base else "",
        "g_start": g_start, "limit_test": limit_test if limit_test else 0,
    }])
    df.to_csv(os.path.join(outdir, "objective_summary.csv"), index=False)
    print(df.to_string(index=False))

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ann_ckpt", required=True)
    ap.add_argument("--rl_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaler_dir", type=str, default=None)
    ap.add_argument("--rl_base", type=str, default=None)
    ap.add_argument("--rl_delta", type=float, default=0.10)
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--g_start", type=int, default=0)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--pc", type=float, default=1e-3)
    ap.add_argument("--limit_test", type=int, default=0)
    args = ap.parse_args()

    evaluate_objective(
        data_path=args.data,
        ann_ckpt=args.ann_ckpt,
        rl_ckpt=args.rl_ckpt,
        outdir=args.out,
        scaler_dir=args.scaler_dir,
        rl_base=args.rl_base,
        rl_delta=args.rl_delta,
        users=args.users,
        g_start=args.g_start,
        sigma2=args.sigma2,
        p_c=args.pc,
        limit_test=args.limit_test,
    )
