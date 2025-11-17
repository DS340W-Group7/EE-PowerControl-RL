# **Energy-Efficient Power Control with ANN + RL (Δ-log-power Fine-Tuning)**

*DS 340W — Simulation-Driven ML for Sustainable Wireless Networks*

This repository implements and extends the energy-efficient power control framework from:

> **Matthiesen et al.,**
> *“A Globally Optimal Energy-Efficient Power Control Framework and its Efficient Implementation in Wireless Interference Networks,”* IEEE TSP.

We train:

1. **ANN baseline** reproducing the original supervised method
2. **RL fine-tuning module** (novelty) that improves the ANN by predicting small Δ-log-power corrections to directly maximize **WSEE** (Weighted Sum Energy Efficiency)

The full dataset (**dset4.h5**) is included inside this repository for easy reproducibility.

---

# 📂 Repository Layout

```
src/                # ANN, RL, environment model, and evaluation scripts
data/               # dataset (included)
results/            # saved models, plots, tables
README.md
requirements.txt
```

---

# ⚙️ Setup

```bash
git clone https://github.com/<your-username>/EE-PowerControl-RL.git
cd EE-PowerControl-RL

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Dataset used:

```
data/dset4.h5   <- included in repo
```

No downloads required.

---

# 🧠 Step 1 — Train ANN Baseline

```bash
python3 src/train_ann.py \
    --data data/dset4.h5 \
    --out results/ann2 \
    --epochs 60 \
    --bs 1024 \
    --lr 5e-4
```

Outputs:

* `results/ann2/model.pt`
* `results/ann2/mu.npy`, `sigma.npy`

---

# 🔁 Step 2 — Train RL (Novelty)

```bash
python3 src/train_rl.py \
    --data data/dset4.h5 \
    --init results/ann2/model.pt \
    --out results/rl3 \
    --users 4 \
    --steps 8000 \
    --bs 512 \
    --lr 1e-5 \
    --g_start 0 \
    --scaler_dir results/ann2 \
    --delta_scale 0.10
```

Outputs:

* `results/rl3/model_rl.pt`

RL learns small Δ-log-power corrections around ANN.

---

# 📈 Step 3 — Objective-Level Comparison (Recommended)

```bash
python3 src/compare_objective.py \
    --data data/dset4.h5 \
    --ann_ckpt results/ann2/model.pt \
    --rl_ckpt  results/rl3/model_rl.pt \
    --out results/obj_compare \
    --scaler_dir results/ann2 \
    --rl_base results/ann2/model.pt \
    --rl_delta 0.10 \
    --users 4 \
    --g_start 0 \
    --limit_test 50000
```

Outputs:

* `cdf_objective_env.png`
* `objective_compare_env.csv`

This is the **main figure** used in the final report.

---

# 📊 Step 4 — Objective Summary Table

```bash
python3 src/eval_objective.py \
    --data data/dset4.h5 \
    --ann_ckpt results/ann2/model.pt \
    --rl_ckpt  results/rl3/model_rl.pt \
    --out results/obj_eval \
    --scaler_dir results/ann2 \
    --rl_base results/ann2/model.pt \
    --rl_delta 0.10 \
    --users 4 \
    --g_start 0 \
    --limit_test 50000
```

Outputs:

* `results/obj_eval/objective_summary.csv`

---

* Best figure:
  `results/obj_compare/cdf_objective_env.png`

* Table summary:
  `results/obj_compare/objective_compare_env.csv`

* Key takeaway:
  **The RL Δ-log-power policy yields significantly higher WSEE than both the ANN baseline and the BB label-based powers.**

---

# 🆘 Troubleshooting

* If RL produces NaNs, ensure you passed `--scaler_dir results/ann2`
* For more RL improvement:

  * Increase steps to 12000
  * Try `--delta_scale 0.15`
* Objective > labels is normal (env uses p_c and σ² consistent with training)
