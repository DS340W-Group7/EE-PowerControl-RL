import argparse, os, numpy as np, torch, matplotlib.pyplot as plt, pandas as pd
from dataio import EEH5Dataset
from models import MLP
from env import unpack_G, rates_log2, wsee_reward, powers_from_policy

def _maybe_load_scaler(sdir):
    if not sdir: return None, None
    mp, sp = os.path.join(sdir,"mu.npy"), os.path.join(sdir,"sigma.npy")
    if os.path.exists(mp) and os.path.exists(sp):
        mu = np.load(mp).astype(np.float32)
        sg = np.load(sp).astype(np.float32)
        sg = np.where(sg < 1e-8, 1e-8, sg).astype(np.float32)
        return mu, sg
    return None, None

def _apply(X, mu, sg): return X if (mu is None or sg is None) else (X - mu)/sg

@torch.no_grad()
def _predict_logp(X, out_dim, ckpt, scaler_dir, base_ckpt=None, delta_scale=0.10):
    mu, sg = _maybe_load_scaler(scaler_dir if scaler_dir else os.path.dirname(ckpt))
    Xn = _apply(X, mu, sg)

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

    return np.clip(Yhat_log, -20.0, 5.0)

def compare_objective(
    data_path, ann_ckpt, rl_ckpt, outdir,
    scaler_dir=None, rl_base=None, rl_delta=0.10,
    users=4, g_start=0, sigma2=1.0, p_c=1e-3, limit_test=0, bs=4096
):
    os.makedirs(outdir, exist_ok=True)
    ds = EEH5Dataset(data_path)
    Xte, Yte, meta = ds.get_split("test")    # Yte: log-powers (labels)
    if limit_test and limit_test < Xte.shape[0]:
        Xte, Yte = Xte[:limit_test], Yte[:limit_test]

    N, out_dim = Yte.shape
    pmax_log = np.max(Yte, axis=1).astype(np.float32)

    # helper to compute per-sample objective
    def wsee_from_logp(logp_np, i0, i1):
        xb   = torch.from_numpy(Xte[i0:i1]).float()
        logp = torch.from_numpy(logp_np[i0:i1].astype(np.float32))
        pmax = torch.from_numpy(pmax_log[i0:i1])
        G = unpack_G(xb, users=users, start=g_start)
        p = powers_from_policy(logp, pmax)
        Ri = rates_log2(G, p, sigma2=sigma2)
        return wsee_reward(Ri, p, p_c=p_c).cpu().numpy()

    # ANN predictions
    ann_log = _predict_logp(Xte, out_dim, ann_ckpt, scaler_dir)
    # RL blended predictions
    rl_log  = _predict_logp(Xte, out_dim, rl_ckpt, scaler_dir, base_ckpt=rl_base, delta_scale=rl_delta)
    # Labels (as "BB powers" in our env path)
    lab_log = np.clip(Yte, -20.0, 5.0)

    # compute WSEE per sample (batched)
    obj_ann = np.concatenate([wsee_from_logp(ann_log, i, min(i+bs,N)) for i in range(0,N,bs)])
    obj_rl  = np.concatenate([wsee_from_logp(rl_log,  i, min(i+bs,N)) for i in range(0,N,bs)])
    obj_lab = np.concatenate([wsee_from_logp(lab_log, i, min(i+bs,N)) for i in range(0,N,bs)])

    # CDF plot (higher is better here)
    def cdf(x): xs = np.sort(x); F = np.linspace(0,1,len(xs)); return xs, F
    xs_a, Fa = cdf(obj_ann)
    xs_r, Fr = cdf(obj_rl)
    xs_b, Fb = cdf(obj_lab)

    plt.figure(figsize=(6,4), dpi=120)
    plt.plot(xs_b, Fb, label="Labels (env)")
    plt.plot(xs_a, Fa, label="ANN (env)")
    plt.plot(xs_r, Fr, label=f"RL (env, blend {rl_delta:.2f})")
    plt.xlabel("WSEE (higher is better)")
    plt.ylabel("CDF")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cdf_objective_env.png"))

    # Summary table
    df = pd.DataFrame([
        {"method":"Labels(env)", "mean_WSEE":float(obj_lab.mean()), "median_WSEE":float(np.median(obj_lab)), "n":N},
        {"method":"ANN(env)",    "mean_WSEE":float(obj_ann.mean()), "median_WSEE":float(np.median(obj_ann)), "n":N},
        {"method":"RL(env)",     "mean_WSEE":float(obj_rl .mean()), "median_WSEE":float(np.median(obj_rl )), "n":N},
    ])
    df.to_csv(os.path.join(outdir, "objective_compare_env.csv"), index=False)
    print(df)

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

    compare_objective(
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
