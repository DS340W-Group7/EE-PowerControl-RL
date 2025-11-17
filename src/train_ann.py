import argparse
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from rich.progress import track

from dataio import EEH5Dataset
from models import MLP


def train(
    data_path: str,
    outdir: str,
    bs: int = 1024,
    epochs: int = 60,
    lr: float = 5e-4,
    seed: int = 0,
    workers: int = 0,
):
    os.makedirs(outdir, exist_ok=True)

    # seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- load splits ----
    ds = EEH5Dataset(data_path)
    Xtr, Ytr, _ = ds.get_split("train")  # Ytr is already LOG-POWER
    Xva, Yva, _ = ds.get_split("val")    # Yva is already LOG-POWER

    # ---- feature scaling (train-only stats) ----
    mu = Xtr.mean(axis=0, keepdims=True)
    sigma = Xtr.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1e-8, sigma)

    Xtr = (Xtr - mu) / sigma
    Xva = (Xva - mu) / sigma

    # save scaler for eval.py
    np.save(os.path.join(outdir, "mu.npy"), mu.astype(np.float32))
    np.save(os.path.join(outdir, "sigma.npy"), sigma.astype(np.float32))

    # labels are already log-power (DO NOT log again)
    Ytr_log = Ytr
    Yva_log = Yva

    # ---- model / opt / loss ----
    model = MLP(in_dim=Xtr.shape[1], out_dim=Ytr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # SmoothL1 is more robust than MSE on log-power targets
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
    #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, verbose=True)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    # ---- loaders ----
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr_log)),
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva_log)),
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )

    best = {"val": float("inf")}
    ckpt_path = os.path.join(outdir, "model.pt")

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        ep_loss = 0.0
        for xb, yb in track(tr_loader, description=f"[train] epoch {ep}", transient=True):
            opt.zero_grad(set_to_none=True)
            pred = model(xb.float())
            loss = loss_fn(pred, yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= len(tr_loader.dataset)

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            va_loss = 0.0
            for xb, yb in va_loader:
                pred = model(xb.float())
                va_loss += loss_fn(pred, yb.float()).item() * xb.size(0)
            va_loss /= len(va_loader.dataset)

        sched.step(va_loss)
        print(f"epoch {ep:02d}: train_smoothL1={ep_loss:.6f}  val_smoothL1={va_loss:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

        # ---- checkpoint ----
        if va_loss < best["val"]:
            best["val"] = va_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"[done] best val loss: {best['val']:.6f}  -> saved to {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dset4.h5")
    ap.add_argument("--out", required=True, help="Output directory (saves model.pt, mu.npy, sigma.npy)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    train(
        data_path=args.data,
        outdir=args.out,
        bs=args.bs,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        workers=args.workers,
    )
