#!/usr/bin/env python3
"""
Section 4.1 linear mixture example.

Baseline: ridge regression fit on n_tr i.i.d. observations from the IC model.
  IC model:  y = m_ic * x + c_ic + ε,  x ~ U(-√3, √3),  ε ~ N(0, noise_sd²)
  OC model:  y = m_oc * x + c_oc + ε  (parameter shift)

Paper defaults (Sec. 4.1):
  n_tr=2000, m_ic=16, c_ic=5, m_oc=12, c_oc=3, noise_sd=4 (σ²=16),
  γ=0.1, λ=0.01, α=0.001, B_O=100, B_I=200.

Two UCL methods are compared:
  1. Nested bootstrap (Algorithm 1) — time-varying UCL with k-correction.
  2. Zhang et al. (two-phase) — constant UCL from T² quantile on D2.

Outputs (default outdir: outputs/linear/):
  trajectory.png         — single monitoring trajectory (Fig. 4.2a analog)
  pfar_comparison.png    — PFAR curves for both methods (Fig. 4.2b analog)
  pfar_boot.{npz,csv}    — bootstrap PFAR results
  pfar_kungang.{npz,csv} — Zhang et al. PFAR results

Usage:
  python conceptdrift_quest/linear_demo.py
  python conceptdrift_quest/linear_demo.py --B-outer 20 --R 200   # quick smoke test
"""
import argparse
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def generate_data(n: int, m: float, c: float, noise_sd: float, rng: np.random.Generator):
    """Draw n i.i.d. observations.

    Returns X_aug (n, 2) with columns [1, x] and y (n,).
    x ~ U(-√3, √3)  →  Var(x) = 1.
    """
    x = rng.uniform(-math.sqrt(3), math.sqrt(3), n)
    y = m * x + c + rng.normal(0.0, noise_sd, n)
    return np.column_stack([np.ones(n), x]), y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def fit_ridge(X: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """Closed-form ridge with no intercept penalty.

    θ̂ = (XᵀX + diag(0, γ))⁻¹ Xᵀy
    """
    p = X.shape[1]
    Gamma = gamma * np.diag(np.r_[0.0, np.ones(p - 1)])
    return np.linalg.solve(X.T @ X + Gamma, X.T @ y)


# ---------------------------------------------------------------------------
# Scores (Eq. 4.3)
# ---------------------------------------------------------------------------
def score_vectors(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    gamma: float,
    n_fit: int,
) -> np.ndarray:
    """Per-observation Fisher score (Eq. 4.3).

    s_i = (y_i − xᵢᵀθ) xᵢ − (γ / n_fit) θ̃,
    where θ̃ = [0, θ₁]ᵀ zeroes the intercept component.

    The zero-mean property Σᵢ sᵢ = 0 holds at the ridge minimiser
    by the KKT stationarity conditions.

    n_fit is the sample size used when fitting theta (controls the penalty
    scale and ensures consistency between training and scoring).
    """
    resid = y - X @ theta
    pen = (gamma / n_fit) * np.r_[0.0, theta[1:]]
    return resid[:, None] * X - pen[None, :]


# ---------------------------------------------------------------------------
# MEWMA T² (Theorem 3.1 / Eq. 3.3)
# ---------------------------------------------------------------------------
def ewma_T2(
    scores: np.ndarray,
    lam: float,
    s_bar: np.ndarray,
    Sigma_inv: np.ndarray,
    n_train: int,
    apply_k_correction: bool,
) -> np.ndarray:
    """MEWMA T² statistic with optional k-correction.

    k(λ, i, n) = num / den  (Eq. 3.3), applied as z̃ᵢ = zᵢ / √k.
    When apply_k_correction=False the correction is skipped (used for
    actual monitoring and the Zhang et al. baseline).
    """
    z = np.zeros_like(s_bar, dtype=float)
    out = np.empty(len(scores))
    for t, s in enumerate(scores, 1):
        z = lam * s + (1.0 - lam) * z
        if apply_k_correction:
            decay = (1.0 - lam) ** t
            var_base = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * t))
            num = var_base + (3.72 / n_train) * (1.0 - decay) ** 2
            den = var_base + (1.0 / n_train) * (1.0 - decay) ** 2
            z_sc = z / math.sqrt(num / den)
        else:
            z_sc = z
        diff = z_sc - s_bar
        out[t - 1] = float(diff @ Sigma_inv @ diff)
    return out


# ---------------------------------------------------------------------------
# Nested bootstrap UCL — Algorithm 1
# ---------------------------------------------------------------------------
def bootstrap_ucl(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    gamma: float,
    lam: float,
    lam_cov: float,
    B_outer: int,
    B_inner: int,
    M: int,
    perc: float,
    rng: np.random.Generator,
):
    """Nested bootstrap control limit (Algorithm 1).

    Returns
    -------
    UCL       : (M,)  time-varying upper control limit
    s_bar     : (p,)  mean score on training data
    Sigma_inv : (p,p) inverse covariance
    theta0    : (p,)  baseline ridge estimate
    t_base    : float  wall time for the baseline fit (seconds)
    """
    n_tr = len(X_tr)
    idx_all = np.arange(n_tr)

    # Baseline fit on full training data
    t0 = time.perf_counter()
    theta0 = fit_ridge(X_tr, y_tr, gamma)
    S_tr = score_vectors(X_tr, y_tr, theta0, gamma, n_fit=n_tr)
    s_bar = S_tr.mean(axis=0)
    Sigma = np.cov(S_tr.T) + lam_cov * np.eye(S_tr.shape[1])
    Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)
    t_base = time.perf_counter() - t0

    res = np.empty((B_outer * B_inner, M))
    for b in range(B_outer):
        idx_boot = rng.choice(n_tr, n_tr, replace=True)
        mask = np.ones(n_tr, dtype=bool)
        mask[np.unique(idx_boot)] = False
        idx_oob = idx_all[mask]

        # Outer-bootstrap model (θ^b)
        theta_b = fit_ridge(X_tr[idx_boot], y_tr[idx_boot], gamma)

        # Scores on the bootstrap draw (n_fit = n_tr, matching the fit)
        S_b = score_vectors(X_tr[idx_boot], y_tr[idx_boot], theta_b, gamma, n_fit=n_tr)
        s_bar_b = S_b.mean(axis=0)
        Sigma_b = np.cov(S_b.T) + lam_cov * np.eye(S_b.shape[1])
        Sigma_b_inv = np.linalg.pinv(Sigma_b, hermitian=True)

        # OOB scores (fall back to boot scores if OOB set is empty)
        if len(idx_oob) == 0:
            S_oob = S_b
        else:
            S_oob = score_vectors(X_tr[idx_oob], y_tr[idx_oob], theta_b, gamma, n_fit=n_tr)

        # Inner bootstrap: resample from OOB to form monitoring streams
        for j in range(B_inner):
            stream = S_oob[rng.choice(len(S_oob), M, replace=True)]
            res[b * B_inner + j] = ewma_T2(
                stream, lam, s_bar_b, Sigma_b_inv,
                n_train=n_tr, apply_k_correction=True,
            )

        if (b + 1) % max(1, B_outer // 5) == 0:
            print(f"  outer bootstrap {b + 1}/{B_outer}")

    UCL = np.percentile(res, perc, axis=0)
    return UCL, s_bar, Sigma_inv, theta0, t_base


# ---------------------------------------------------------------------------
# Zhang et al. two-phase UCL
# ---------------------------------------------------------------------------
def kungang_ucl(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    gamma: float,
    lam: float,
    lam_cov: float,
    perc: float,
):
    """Zhang et al. two-phase method.

    Splits training data in half: D1 for model fit, D2 for UCL calibration.

    Returns
    -------
    UCL_const : float  constant upper control limit
    mu        : (p,)  mean score on D2
    Sigma_inv : (p,p) inverse covariance (estimated from D1)
    theta_D1  : (p,)  ridge estimate from D1
    n_D1      : int   size of D1 (used as n_fit for monitoring scores)
    """
    n = len(X_tr)
    n_half = n // 2
    X_D1, y_D1 = X_tr[:n_half], y_tr[:n_half]
    X_D2, y_D2 = X_tr[n_half:], y_tr[n_half:]

    theta_D1 = fit_ridge(X_D1, y_D1, gamma)

    S_D1 = score_vectors(X_D1, y_D1, theta_D1, gamma, n_fit=n_half)
    Sigma = np.cov(S_D1.T) + lam_cov * np.eye(S_D1.shape[1])
    Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)

    S_D2 = score_vectors(X_D2, y_D2, theta_D1, gamma, n_fit=n_half)
    mu = S_D2.mean(axis=0)
    T2_D2 = ewma_T2(S_D2, lam, mu, Sigma_inv, n_train=n_half, apply_k_correction=False)
    UCL_const = float(np.percentile(T2_D2, perc))

    return UCL_const, mu, Sigma_inv, theta_D1, n_half


# ---------------------------------------------------------------------------
# PFAR estimation
# ---------------------------------------------------------------------------
def estimate_pfar(
    n_ic: int,
    n_oc: int,
    m_ic: float,
    c_ic: float,
    m_oc: float,
    c_oc: float,
    noise_sd: float,
    theta: np.ndarray,
    s_bar: np.ndarray,
    Sigma_inv: np.ndarray,
    UCL,               # scalar (Kungang) or array (bootstrap)
    gamma: float,
    lam: float,
    n_fit: int,
    R: int,
    rng: np.random.Generator,
    label: str = "",
) -> np.ndarray:
    """Monte Carlo estimate of pointwise FAR over n_ic IC + n_oc OC obs.

    PFAR[i] = P(T²_i > UCL_i | process state at i).
    """
    M = n_ic + n_oc
    exceed = np.zeros(M, dtype=int)
    for r in range(R):
        X_ic, y_ic = generate_data(n_ic, m_ic, c_ic, noise_sd, rng)
        X_oc, y_oc = generate_data(n_oc, m_oc, c_oc, noise_sd, rng)
        X_mon = np.vstack([X_ic, X_oc])
        y_mon = np.hstack([y_ic, y_oc])
        S = score_vectors(X_mon, y_mon, theta, gamma, n_fit=n_fit)
        T2 = ewma_T2(S, lam, s_bar, Sigma_inv, n_train=n_fit, apply_k_correction=False)
        exceed += (T2 > UCL).astype(int)
        if (r + 1) % max(1, R // 5) == 0:
            print(f"  [{label}] PFAR {r + 1}/{R}")
    return exceed / R


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_trajectory(
    T2: np.ndarray,
    UCL: np.ndarray,
    UCL_const: float,
    n_ic: int,
    outdir: str,
):
    """Single monitoring trajectory showing both UCLs and the change-point."""
    M = len(T2)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(np.arange(M), T2, lw=1.5, color="#1f77b4", label=r"MEWMA $T^2_i$")
    ax.semilogy(np.arange(M), UCL, lw=2, ls="--", color="#2ca02c", label="Bootstrap UCL")
    ax.axhline(UCL_const, lw=2, ls=":", color="#d95f02", label="Zhang et al. UCL")
    ax.axvline(n_ic - 0.5, color="k", lw=1.5, ls="--", label="Change-point")
    ax.set_xlabel("Sample index $i$")
    ax.set_ylabel(r"$T^2_i$ (log scale)")
    ax.set_title("Monitoring Trajectory — Linear Example (Sec. 4.1)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=13)
    plt.tight_layout()
    path = os.path.join(outdir, "trajectory.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_pfar_comparison(
    pfar_boot: np.ndarray,
    pfar_kg: np.ndarray,
    n_ic: int,
    alpha_nom: float,
    outdir: str,
):
    """PFAR comparison: bootstrap (ours) vs. Zhang et al."""
    M = len(pfar_boot)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(np.arange(M), pfar_boot, lw=2, color="#2ca02c", label="Bootstrap (ours)")
    ax.semilogy(np.arange(M), pfar_kg, lw=2, ls="--", color="#d95f02", label="Zhang et al.")
    ax.axhline(alpha_nom, lw=2, ls=":", color="#7f7f7f",
               label=f"Nominal $\\alpha={alpha_nom}$")
    ax.axvline(n_ic - 0.5, color="k", lw=1.5, ls="--", label="Change-point")
    ax.set_xlabel("Sample index $i$")
    ax.set_ylabel("Pointwise FAR (log scale)")
    ax.set_title("PFAR Comparison — Linear Example (Sec. 4.1)")
    ax.set_ylim(alpha_nom * 0.01, 1.0)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "pfar_comparison.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Section 4.1 linear example: bootstrap vs. Zhang et al."
    )
    ap.add_argument("--n-tr", dest="n_tr", type=int, default=2000,
                    help="Baseline training sample size")
    ap.add_argument("--m-ic", dest="m_ic", type=float, default=16.0,
                    help="IC slope (model 4.1)")
    ap.add_argument("--c-ic", dest="c_ic", type=float, default=5.0,
                    help="IC intercept (model 4.1)")
    ap.add_argument("--m-oc", dest="m_oc", type=float, default=12.0,
                    help="OC slope (model 4.2)")
    ap.add_argument("--c-oc", dest="c_oc", type=float, default=3.0,
                    help="OC intercept (model 4.2)")
    ap.add_argument("--noise-sd", dest="noise_sd", type=float, default=4.0,
                    help="Noise std dev sigma (sigma^2=16 in the paper)")
    ap.add_argument("--gamma", type=float, default=0.1,
                    help="Ridge penalty gamma")
    ap.add_argument("--lam", type=float, default=0.01,
                    help="MEWMA smoothing parameter lambda")
    ap.add_argument("--lam-cov", dest="lam_cov", type=float, default=1e-8,
                    help="Covariance ridge regularization (small: scores are 2-D)")
    ap.add_argument("--B-outer", dest="B_outer", type=int, default=100,
                    help="Outer bootstrap replicates B_O")
    ap.add_argument("--B-inner", dest="B_inner", type=int, default=200,
                    help="Inner bootstrap replicates B_I")
    ap.add_argument("--perc", type=float, default=99.9,
                    help="UCL percentile (99.9 -> alpha ~ 0.001)")
    ap.add_argument("--n-ic", dest="n_ic", type=int, default=200,
                    help="IC observations in each monitoring stream")
    ap.add_argument("--n-oc", dest="n_oc", type=int, default=800,
                    help="OC observations in each monitoring stream")
    ap.add_argument("--R", type=int, default=2000,
                    help="Monte Carlo replicates for PFAR estimation")
    ap.add_argument("--alpha-nom", dest="alpha_nom", type=float, default=0.001,
                    help="Nominal false alarm rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="outputs/linear")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    M = args.n_ic + args.n_oc

    # ---- Training data ----
    print(f"Generating {args.n_tr} training observations (IC model: "
          f"y = {args.m_ic}x + {args.c_ic} + N(0, {args.noise_sd}^2))...")
    X_tr, y_tr = generate_data(args.n_tr, args.m_ic, args.c_ic, args.noise_sd, rng)

    # ---- Bootstrap UCL ----
    print(f"\nNested bootstrap UCL (B_outer={args.B_outer}, B_inner={args.B_inner})...")
    t0 = time.perf_counter()
    UCL, s_bar, Sigma_inv, theta0, t_base = bootstrap_ucl(
        X_tr, y_tr, args.gamma, args.lam, args.lam_cov,
        args.B_outer, args.B_inner, M, args.perc, rng,
    )
    t_boot_wall = time.perf_counter() - t0
    print(f"Bootstrap done in {t_boot_wall:.1f}s (baseline fit {t_base:.3f}s)")

    # ---- Zhang et al. UCL ----
    print("\nComputing Zhang et al. (two-phase) UCL...")
    UCL_const, mu_kg, Sigma_inv_kg, theta_D1, n_D1 = kungang_ucl(
        X_tr, y_tr, args.gamma, args.lam, args.lam_cov, args.perc,
    )
    print(f"Kungang UCL_const = {UCL_const:.4f}")

    # ---- Single monitoring trajectory (Fig. 4.2a) ----
    print("\nGenerating single monitoring trajectory...")
    X_ic, y_ic = generate_data(args.n_ic, args.m_ic, args.c_ic, args.noise_sd, rng)
    X_oc, y_oc = generate_data(args.n_oc, args.m_oc, args.c_oc, args.noise_sd, rng)
    S_mon = score_vectors(
        np.vstack([X_ic, X_oc]), np.hstack([y_ic, y_oc]),
        theta0, args.gamma, n_fit=args.n_tr,
    )
    T2_traj = ewma_T2(
        S_mon, args.lam, s_bar, Sigma_inv,
        n_train=args.n_tr, apply_k_correction=False,
    )
    plot_trajectory(T2_traj, UCL, UCL_const, args.n_ic, args.outdir)

    # ---- PFAR estimation (Fig. 4.2b) ----
    print(f"\nEstimating PFAR (R={args.R} MC replicates each)...")
    pfar_boot = estimate_pfar(
        args.n_ic, args.n_oc,
        args.m_ic, args.c_ic, args.m_oc, args.c_oc, args.noise_sd,
        theta0, s_bar, Sigma_inv, UCL,
        args.gamma, args.lam, args.n_tr, args.R, rng, label="boot",
    )
    pfar_kg = estimate_pfar(
        args.n_ic, args.n_oc,
        args.m_ic, args.c_ic, args.m_oc, args.c_oc, args.noise_sd,
        theta_D1, mu_kg, Sigma_inv_kg, UCL_const,
        args.gamma, args.lam, n_D1, args.R, rng, label="kungang",
    )
    plot_pfar_comparison(pfar_boot, pfar_kg, args.n_ic, args.alpha_nom, args.outdir)

    # ---- Save results ----
    np.savez(
        os.path.join(args.outdir, "pfar_boot.npz"),
        pfar=pfar_boot, UCL=UCL, s_bar=s_bar, theta=theta0,
        n_ic=args.n_ic, n_oc=args.n_oc, R=args.R,
        t_boot_wall=t_boot_wall, t_base=t_base,
        params=dict(
            n_tr=args.n_tr, B_outer=args.B_outer, B_inner=args.B_inner,
            gamma=args.gamma, lam=args.lam, perc=args.perc,
            noise_sd=args.noise_sd, m_ic=args.m_ic, c_ic=args.c_ic,
        ),
    )
    np.savez(
        os.path.join(args.outdir, "pfar_kungang.npz"),
        pfar=pfar_kg, UCL_const=UCL_const, mu=mu_kg, theta_D1=theta_D1,
        n_ic=args.n_ic, n_oc=args.n_oc, R=args.R, n_D1=n_D1,
        params=dict(
            n_tr=args.n_tr, gamma=args.gamma, lam=args.lam, perc=args.perc,
            noise_sd=args.noise_sd, m_ic=args.m_ic, c_ic=args.c_ic,
        ),
    )
    np.savetxt(
        os.path.join(args.outdir, "pfar_boot.csv"),
        np.c_[np.arange(M), pfar_boot],
        delimiter=",", header="i,PFAR", comments="",
    )
    np.savetxt(
        os.path.join(args.outdir, "pfar_kungang.csv"),
        np.c_[np.arange(M), pfar_kg],
        delimiter=",", header="i,PFAR", comments="",
    )

    print(f"\nAll outputs written to {args.outdir}/")
    print(f"Boot    PFAR: IC mean={pfar_boot[:args.n_ic].mean():.4f}  "
          f"OC mean={pfar_boot[args.n_ic:].mean():.4f}")
    print(f"Kungang PFAR: IC mean={pfar_kg[:args.n_ic].mean():.4f}  "
          f"OC mean={pfar_kg[args.n_ic:].mean():.4f}")


if __name__ == "__main__":
    main()
