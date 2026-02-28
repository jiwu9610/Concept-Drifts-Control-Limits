#!/usr/bin/env python3
"""
Quick, single-node PFAR demonstration for the nonlinear coupled oscillators.
- Computes bootstrap UCL (nested bootstrap) for each requested noise level.
- Runs Monte Carlo monitoring streams to estimate PFAR and produces a plot.

Defaults are trimmed for speed; increase B_outer/B_inner/R for tighter FAR
estimates (alpha=0.001 target). Alpha=0.01 is acceptable if time is tight.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import matplotlib.pyplot as plt

RNG = np.random.default_rng(16)
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


# -------------------------
# Simulator / data
# -------------------------
def simulate_system(t_grid, init, p):
    def f(state, _t, m1, m2, k1, k2, k3, c1, c2):
        p1, v1, p2, v2 = state
        delta = p1 - p2
        phi = delta / (1 + abs(delta))
        return [
            v1,
            (-k1 * p1 - c1 * v1 + k3 * phi) / m1,
            v2,
            (-k2 * p2 - c2 * v2 - k3 * phi) / m2,
        ]

    args = tuple(p[k] for k in ("m1", "m2", "k1", "k2", "k3", "c1", "c2"))
    return odeint(f, init, t_grid, args=args)


def default_init_sampler():
    base = np.array([0.5, 0.0, -0.5, 0.0], dtype=float)
    jitter = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    return base + RNG.uniform(-1.0, 1.0, size=4) * jitter


def energy_from_state(p1, v1, p2, v2, params):
    delta = p1 - p2
    phi = delta / (1 + np.abs(delta))
    return (
        0.5 * (params["m1"] * v1**2 + params["m2"] * v2**2)
        + 0.5 * (params["k1"] * p1**2 + params["k2"] * p2**2)
        + params["k3"] * phi
    )


def sample_one_trajectory_fixed_time(params, noise_sd, *, t_max=30.0, T=20, t_star=12.0, init_sampler=None):
    if init_sampler is None:
        init_sampler = default_init_sampler
    init = init_sampler()
    t_grid = np.linspace(0.0, t_max, T)
    sol = simulate_system(t_grid, init, params)
    p1, v1, p2, v2 = sol.T
    X = np.column_stack([p1, v1, p2, v2]).ravel()
    j_star = int(np.clip(np.round((t_star / t_max) * (T - 1)), 0, T - 1))
    p1s, v1s, p2s, v2s = sol[j_star]
    y_det = float(energy_from_state(p1s, v1s, p2s, v2s, params))
    y = y_det + float(RNG.normal(0.0, noise_sd))
    return X, y


def generate_iid_trajectories(n_samples, params, *, noise_sd=0.03, t_max=30.0, T=20, t_star=12.0, init_sampler=None):
    X = np.empty((n_samples, 4 * T), dtype=float)
    y = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        Xi, yi = sample_one_trajectory_fixed_time(
            params, noise_sd, t_max=t_max, T=T, t_star=t_star, init_sampler=init_sampler
        )
        X[i] = Xi
        y[i] = yi
    return X, y


# -------------------------
# Model / scores
# -------------------------
class MLP(nn.Module):
    def __init__(self, p_in, hidden=32, depth=4):
        super().__init__()
        layers = [nn.Linear(p_in, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_lastlayer_penalty_only(model, X, y, alpha_lastlayer, epochs=3000, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    last = last_linear_layer(model)
    W, b = last.weight, last.bias
    n = X.shape[0]
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(X).squeeze()
        resid = pred - y
        loss = 0.5 * (resid.square().mean()) + (alpha_lastlayer / (2.0 * n)) * (
            W.square().sum() + b.square().sum()
        )
        loss.backward()
        opt.step()
    return model


def last_linear_layer(model: nn.Module) -> nn.Linear:
    last = model.net[-1]
    assert isinstance(last, nn.Linear)
    return last


def score_vectors_last_layer(X, y, model, alpha_lastlayer):
    last = last_linear_layer(model)
    W = last.weight
    b = last.bias
    grads = []
    n = X.shape[0]
    for xi, yi in zip(X, y):
        model.zero_grad(set_to_none=True)
        pred = model(xi[None]).squeeze()
        resid = pred - yi
        loss = 0.5 * resid.square() + (alpha_lastlayer / (2.0 * n)) * (
            W.square().sum() + b.square().sum()
        )
        loss.backward()
        grads.append(torch.cat([W.grad.flatten(), b.grad.flatten()]))
    return torch.stack(grads).detach().cpu().numpy()


# -------------------------
# EWMA T2
# -------------------------
def ewma_T2(stream_scores, lam, s_bar, Sigma_inv, n_train, apply_k_correction: bool):
    z = np.zeros_like(s_bar, dtype=float)
    out = np.empty(len(stream_scores), dtype=float)
    for t, s in enumerate(stream_scores, 1):
        z = lam * s + (1 - lam) * z
        if apply_k_correction:
            num = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t)) + (
                3.72 / n_train
            ) * (1 - (1 - lam) ** t) ** 2
            den = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t)) + (
                1.0 / n_train
            ) * (1 - (1 - lam) ** t) ** 2
            z_scaled = z / np.sqrt(num / den)
        else:
            z_scaled = z
        diff = z_scaled - s_bar
        out[t - 1] = diff @ Sigma_inv @ diff
    return out


# -------------------------
# Bootstrap UCL (single node)
# -------------------------
def bootstrap_ucl_single(
    X_tr, y_tr, model_fn, *, alpha_lastlayer, lam, lam_cov, B_outer, B_inner, M, perc, fit_epochs_outer, fit_lr_outer
):
    device = X_tr.device
    n_tr = len(X_tr)
    idx_all = np.arange(n_tr)

    t0 = time.perf_counter()
    model0 = train_lastlayer_penalty_only(
        model_fn().to(device),
        X_tr,
        y_tr,
        alpha_lastlayer=alpha_lastlayer,
        epochs=3000,
        lr=1e-3,
    )
    S_tr = score_vectors_last_layer(X_tr, y_tr, model0, alpha_lastlayer)
    s_bar = S_tr.mean(axis=0)
    Sigma = np.cov(S_tr.T) + lam_cov * np.eye(S_tr.shape[1])
    Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)
    t_base = time.perf_counter() - t0

    res = np.empty((B_outer * B_inner, M), dtype=float)
    for b in range(B_outer):
        idx_boot = RNG.choice(n_tr, n_tr, replace=True)
        mask = np.ones(n_tr, dtype=bool)
        mask[np.unique(idx_boot)] = False
        idx_oob = idx_all[mask]

        mdl_b = train_lastlayer_penalty_only(
            model_fn().to(device),
            X_tr[idx_boot],
            y_tr[idx_boot],
            alpha_lastlayer=alpha_lastlayer,
            epochs=fit_epochs_outer,
            lr=fit_lr_outer,
        )

        S_b = score_vectors_last_layer(X_tr[idx_boot], y_tr[idx_boot], mdl_b, alpha_lastlayer)
        s_bar_b = S_b.mean(axis=0)
        Sigma_b = np.cov(S_b.T) + lam_cov * np.eye(S_b.shape[1])
        Sigma_b_inv = np.linalg.pinv(Sigma_b, hermitian=True)

        if len(idx_oob) == 0:
            S_oob = S_b
        else:
            S_oob = score_vectors_last_layer(X_tr[idx_oob], y_tr[idx_oob], mdl_b, alpha_lastlayer)

        for j in range(B_inner):
            idx_inner = RNG.choice(len(S_oob), M, replace=True)
            stream = S_oob[idx_inner]
            res[b * B_inner + j] = ewma_T2(
                stream, lam, s_bar_b, Sigma_b_inv, n_train=n_tr, apply_k_correction=True
            )

    UCL = np.percentile(res, perc, axis=0)
    return UCL, s_bar, Sigma_inv, model0, t_base


# -------------------------
# Monitoring + PFAR
# -------------------------
def run_monitoring_for_noise(noise_sd, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = dict(m1=1.0, m2=2.0, k1=1.0, k2=2.0, k3=1.5, c1=0.1, c2=0.2)
    shift = {**base, "m1": 1.1 * base["m1"], "m2": 1.2 * base["m2"], "k1": 1.3 * base["k1"]}

    n_tr, n_ic, n_oc = args.n_tr, args.n_ic, args.n_oc
    M = n_ic + n_oc

    X_tr, y_tr = generate_iid_trajectories(
        n_tr, base, noise_sd=noise_sd, t_max=args.t_max, T=args.T, t_star=args.t_star
    )
    X_tr_s = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_s = torch.as_tensor(y_tr, dtype=torch.float32, device=device)

    model_fn = lambda: MLP(X_tr_s.shape[1], hidden=args.hidden, depth=args.depth)

    print(f"[noise {noise_sd}] bootstrapping UCL (B_outer={args.B_outer}, B_inner={args.B_inner})…")
    t0 = time.perf_counter()
    UCL, s_bar, Sigma_inv, model0, t_base = bootstrap_ucl_single(
        X_tr_s,
        y_tr_s,
        model_fn,
        alpha_lastlayer=args.alpha_lastlayer,
        lam=args.lam,
        lam_cov=args.lam_cov,
        B_outer=args.B_outer,
        B_inner=args.B_inner,
        M=M,
        perc=args.perc,
        fit_epochs_outer=args.fit_epochs_outer,
        fit_lr_outer=args.fit_lr_outer,
    )
    t_boot_wall = time.perf_counter() - t0
    print(f"[noise {noise_sd}] baseline+UCL done in {t_boot_wall:.1f}s (baseline fit {t_base:.1f}s)")

    # Monte Carlo PFAR
    exceed = np.zeros(M, dtype=int)
    t_pfar0 = time.perf_counter()
    for r in range(args.R):
        X_ic, y_ic = generate_iid_trajectories(
            n_ic, base, noise_sd=noise_sd, t_max=args.t_max, T=args.T, t_star=args.t_star
        )
        X_oc, y_oc = generate_iid_trajectories(
            n_oc, shift, noise_sd=noise_sd, t_max=args.t_max, T=args.T, t_star=args.t_star
        )
        X_mon = np.vstack([X_ic, X_oc])
        y_mon = np.hstack([y_ic, y_oc])
        X_mon_s = X_mon.astype(np.float32)
        y_mon_s = y_mon.astype(np.float32)
        X_t = torch.as_tensor(X_mon_s, device=device)
        y_t = torch.as_tensor(y_mon_s, device=device)
        S = score_vectors_last_layer(X_t, y_t, model0, args.alpha_lastlayer)
        T2 = ewma_T2(S, args.lam, s_bar, Sigma_inv, n_train=len(X_tr), apply_k_correction=False)
        exceed += (T2 > UCL).astype(int)
        if (r + 1) % max(1, args.R // 5) == 0:
            print(f"[noise {noise_sd}] PFAR progress {r+1}/{args.R}")
    t_pfar_wall = time.perf_counter() - t_pfar0

    pfar = exceed / args.R
    return UCL, pfar, M, n_ic, t_boot_wall, t_base, t_pfar_wall


def plot_pfar(UCL, pfar, M, n_ic, noise_sd, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(np.arange(M), pfar, label=f"Empirical PFAR (noise={noise_sd})", lw=2)
    ax.axhline(0.001, ls="--", color="#d95f02", lw=2, label="alpha=0.001")
    ax.axvline(n_ic - 0.5, color="k", lw=1.3, ls="--", label="change-point")
    ax.set_xlabel("Sample index i")
    ax.set_ylabel("PFAR (log scale)")
    ax.set_title("Pointwise False Alarm Rate")
    ax.set_ylim(1e-5, 1e0)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"pfar_noise{noise_sd}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    print(f"[noise {noise_sd}] wrote {fname}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.03, 0.20])
    ap.add_argument("--R", type=int, default=1000, help="Monte Carlo runs per noise level for PFAR")
    ap.add_argument("--B-outer", dest="B_outer", type=int, default=50)
    ap.add_argument("--B-inner", dest="B_inner", type=int, default=200)
    ap.add_argument("--n-tr", dest="n_tr", type=int, default=3000)
    ap.add_argument("--n-ic", dest="n_ic", type=int, default=200)
    ap.add_argument("--n-oc", dest="n_oc", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--alpha-lastlayer", dest="alpha_lastlayer", type=float, default=1e-1)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--lam-cov", dest="lam_cov", type=float, default=1e-1)
    ap.add_argument("--perc", type=float, default=99.9)
    ap.add_argument("--fit-epochs-outer", dest="fit_epochs_outer", type=int, default=500)
    ap.add_argument("--fit-lr-outer", dest="fit_lr_outer", type=float, default=1e-3)
    ap.add_argument("--fit-epochs-base", dest="fit_epochs_base", type=int, default=3000)
    ap.add_argument("--fit-lr-base", dest="fit_lr_base", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--t-max", dest="t_max", type=float, default=30.0)
    ap.add_argument("--t-star", dest="t_star", type=float, default=12.0)
    ap.add_argument("--outdir", default="outputs/pfar_quick")
    args = ap.parse_args()

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    for noise_sd in args.noise:
        UCL, pfar, M, n_ic, t_boot_wall, t_base, t_pfar_wall = run_monitoring_for_noise(noise_sd, args)
        base = os.path.join(args.outdir, f"pfar_noise{noise_sd}")
        os.makedirs(args.outdir, exist_ok=True)
        np.savez(
            base + ".npz",
            pfar=pfar,
            UCL=UCL,
            n_ic=n_ic,
            n_oc=args.n_oc,
            R=args.R,
            t_boot_wall=t_boot_wall,
            t_base_fit=t_base,
            t_pfar_wall=t_pfar_wall,
            params=dict(
                B_outer=args.B_outer,
                B_inner=args.B_inner,
                n_tr=args.n_tr,
                lam=args.lam,
                lam_cov=args.lam_cov,
                perc=args.perc,
                alpha_lastlayer=args.alpha_lastlayer,
                hidden=args.hidden,
                depth=args.depth,
                noise_sd=noise_sd,
                fit_epochs_outer=args.fit_epochs_outer,
                fit_epochs_base=args.fit_epochs_base,
                fit_lr_outer=args.fit_lr_outer,
                fit_lr_base=args.fit_lr_base,
            ),
        )
        np.savetxt(base + ".csv", np.c_[np.arange(M), pfar], delimiter=",", header="i,PFAR", comments="")
        plot_pfar(UCL, pfar, M, n_ic, noise_sd, args.outdir)
        print(f"[noise {noise_sd}] PFAR mean={pfar.mean():.4f}, max={pfar.max():.4f}")


if __name__ == "__main__":
    main()
