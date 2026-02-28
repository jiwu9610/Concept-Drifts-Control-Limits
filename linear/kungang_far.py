#!/usr/bin/env python3
"""
Kungang Zhang et al. (2023) two-phase EWMA (constant UCL) PFAR evaluation.
We split stable data into D1/D2, fit model on D1, build UCL from T2 on D2,
then estimate PFAR on future in-control streams. Supports multiple noise levels.
"""
import argparse
import os
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


def generate_null_data(seed: int, n: int, m: float = 16.0, c: float = 5.0, noise_sd: float = 1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-math.sqrt(3), math.sqrt(3), n)  # Var(x)=1
    y = m * x + c + rng.normal(0, noise_sd, n)
    return x, y


def score_vectors(X: np.ndarray, y: np.ndarray, model: Ridge, alpha: float):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    beta = np.append(model.intercept_, model.coef_)
    resid = y - X_aug @ beta
    reg = 2 * alpha * np.append(0, model.coef_) / X.shape[0]
    return (-2 * resid)[:, None] * X_aug + reg


def ewma_T2(S: np.ndarray, lam: float, mu: np.ndarray, Sigma_inv: np.ndarray):
    z = np.zeros_like(mu)
    out = np.empty(len(S))
    for t, s in enumerate(S):
        z = lam * s + (1 - lam) * z
        diff = (z - mu)[:, None]
        out[t] = (diff.T @ Sigma_inv @ diff).item()
    return out


def process_db_kungang(seed: int, cfg: dict):
    # split stable data
    x, y = generate_null_data(seed, cfg["n_total"], noise_sd=cfg["noise_sd"])
    n_half = cfg["n_total"] // 2
    x_D1, y_D1 = x[:n_half], y[:n_half]  # training
    x_D2, y_D2 = x[n_half:], y[n_half:]  # Phase-I for mu & UCL

    # fit model on D1
    scaler = StandardScaler()
    X_D1 = scaler.fit_transform(x_D1[:, None])
    model = Ridge(alpha=cfg["alpha"]).fit(X_D1, y_D1)

    # Sigma from D1
    S_D1 = score_vectors(X_D1, y_D1, model, cfg["alpha"])
    Sigma = np.cov(S_D1.T) + cfg["lam_cov"] * np.eye(S_D1.shape[1])
    Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)

    # mu and UCL from D2
    X_D2 = scaler.transform(x_D2[:, None])
    S_D2 = score_vectors(X_D2, y_D2, model, cfg["alpha"])
    mu = S_D2.mean(axis=0)
    T2_D2 = ewma_T2(S_D2, cfg["lam"], mu, Sigma_inv)
    UCL = np.percentile(T2_D2, cfg["perc"])

    # future IC streams to estimate FAR
    exc = np.zeros((cfg["N_future"], cfg["M"]))
    for j in range(cfg["N_future"]):
        seed_j = seed * cfg["N_future"] + j
        xf, yf = generate_null_data(seed_j, cfg["M"], noise_sd=cfg["noise_sd"])
        Xf = scaler.transform(xf[:, None])
        Sf = score_vectors(Xf, yf, model, cfg["alpha"])
        T2_f = ewma_T2(Sf, cfg["lam"], mu, Sigma_inv)
        exc[j] = (T2_f > UCL)
    return exc


def evaluate_far_kungang(cfg: dict, N_db: int, max_workers: int):
    all_exc = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(process_db_kungang, seed, cfg) for seed in range(N_db)]
        for fut in as_completed(futs):
            all_exc.append(fut.result())
    all_exc = np.asarray(all_exc)  # shape (db, rep, M)
    far = all_exc.mean(axis=(0, 1))
    return far


def plot_far(far: np.ndarray, alpha_nom: float, out_png: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(np.arange(len(far)), far, lw=2, label="Empirical FAR", color="#1f77b4")
    ax.axhline(alpha_nom, ls="--", lw=2, color="#d95f02", label=f"Nominal α={alpha_nom}")
    ax.set_xlabel("Sample index i")
    ax.set_ylabel("False-Alarm Rate (log scale)")
    ax.set_title("Pointwise False Alarm Rate (Zhang et al.)")
    ax.set_ylim(alpha_nom * 1e-2, 1)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.03, 0.25])
    ap.add_argument("--n-total", dest="n_total", type=int, default=2000)
    ap.add_argument("--M", type=int, default=1000)
    ap.add_argument("--N-future", dest="N_future", type=int, default=1000)
    ap.add_argument("--N-db", dest="N_db", type=int, default=50)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--lam-cov", dest="lam_cov", type=float, default=1e-1)
    ap.add_argument("--perc", type=float, default=99.9)
    ap.add_argument("--alpha-nom", type=float, default=0.001)
    ap.add_argument("--outdir", default="outputs/kungang_far")
    ap.add_argument("--max-workers", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    workers = args.max_workers or max(1, multiprocessing.cpu_count() - 1)

    print(f"Using up to {workers} workers; noise list = {args.noise}")
    for noise_sd in args.noise:
        cfg = dict(
            n_total=args.n_total,
            M=args.M,
            N_future=args.N_future,
            lam=args.lam,
            alpha=args.alpha,
            lam_cov=args.lam_cov,
            perc=args.perc,
            noise_sd=noise_sd,
        )
        print(f"[noise {noise_sd}] evaluating FAR with N_db={args.N_db}, N_future={args.N_future}, M={args.M}")
        far = evaluate_far_kungang(cfg, N_db=args.N_db, max_workers=workers)
        base = f"kungang_noise{noise_sd}"
        out_npz = os.path.join(args.outdir, f"{base}.npz")
        out_csv = os.path.join(args.outdir, f"{base}.csv")
        out_png = os.path.join(args.outdir, f"{base}.png")
        np.savez(out_npz, i=np.arange(len(far)), FAR=far, cfg=cfg)
        pd.DataFrame({"i": np.arange(len(far)), "FAR": far}).to_csv(out_csv, index=False)
        plot_far(far, args.alpha_nom, out_png)
        print(f"[noise {noise_sd}] wrote {out_npz}, {out_csv}, {out_png}")


if __name__ == "__main__":
    main()
