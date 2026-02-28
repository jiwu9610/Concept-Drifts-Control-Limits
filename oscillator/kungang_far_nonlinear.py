#!/usr/bin/env python3
"""
Kungang (two-phase) method evaluated on the nonlinear oscillator example.
- Split baseline data into D1 (fit) and D2 (UCL from T2 on D2).
- Use constant UCL (99.9% unless overridden).
- Estimate PFAR on monitoring streams with a change-point (IC -> OC) to mirror our method.
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
def ewma_T2(stream_scores, lam, mu, Sigma_inv):
    z = np.zeros_like(mu, dtype=float)
    out = np.empty(len(stream_scores), dtype=float)
    for t, s in enumerate(stream_scores, 1):
        z = lam * s + (1 - lam) * z
        diff = z - mu
        out[t - 1] = diff @ Sigma_inv @ diff
    return out


def fit_two_phase(args, noise_sd, device):
    base = dict(m1=1.0, m2=2.0, k1=1.0, k2=2.0, k3=1.5, c1=0.1, c2=0.2)
    n_total = args.n_total
    n_half = n_total // 2
    X_all, y_all = generate_iid_trajectories(
        n_total, base, noise_sd=noise_sd, t_max=args.t_max, T=args.T, t_star=args.t_star
    )

    # split D1/D2
    X_D1, y_D1 = X_all[:n_half], y_all[:n_half]
    X_D2, y_D2 = X_all[n_half:], y_all[n_half:]

    X_D1_t = torch.as_tensor(X_D1.astype(np.float32), device=device)
    y_D1_t = torch.as_tensor(y_D1.astype(np.float32), device=device)

    model_fn = lambda: MLP(X_D1_t.shape[1], hidden=args.hidden, depth=args.depth)
    t0 = time.perf_counter()
    model = train_lastlayer_penalty_only(
        model_fn().to(device),
        X_D1_t,
        y_D1_t,
        alpha_lastlayer=args.alpha_lastlayer,
        epochs=args.fit_epochs,
        lr=args.fit_lr,
    )
    t_fit = time.perf_counter() - t0

    # Sigma from D1
    S_D1 = score_vectors_last_layer(X_D1_t, y_D1_t, model, args.alpha_lastlayer)
    Sigma = np.cov(S_D1.T) + args.lam_cov * np.eye(S_D1.shape[1])
    Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)

    # mu and UCL from D2
    X_D2_t = torch.as_tensor(X_D2.astype(np.float32), device=device)
    y_D2_t = torch.as_tensor(y_D2.astype(np.float32), device=device)
    S_D2 = score_vectors_last_layer(X_D2_t, y_D2_t, model, args.alpha_lastlayer)
    mu = S_D2.mean(axis=0)
    T2_D2 = ewma_T2(S_D2, args.lam, mu, Sigma_inv)
    UCL = np.percentile(T2_D2, args.perc)

    return dict(
        model=model,
        mu=mu,
        Sigma_inv=Sigma_inv,
        UCL=UCL,
        t_fit=t_fit,
    )


def pfar_with_change(model_bundle, args, noise_sd, device):
    base = dict(m1=1.0, m2=2.0, k1=1.0, k2=2.0, k3=1.5, c1=0.1, c2=0.2)
    shift = {**base, "m1": 1.1 * base["m1"], "m2": 1.2 * base["m2"], "k1": 1.3 * base["k1"]}
    n_ic, n_oc = args.n_ic, args.n_oc
    M = n_ic + n_oc

    mu, Sigma_inv, UCL = model_bundle["mu"], model_bundle["Sigma_inv"], model_bundle["UCL"]
    model = model_bundle["model"]

    exceed = np.zeros(M, dtype=int)

    for r in range(args.R):
        X_ic, y_ic = generate_iid_trajectories(
            n_ic, base, noise_sd=noise_sd, t_max=args.t_max, T=args.T, t_star=args.t_star
        )
        X_oc, y_oc = generate_iid_trajectories(
            n_oc, shift, noise_sd=noise_sd, t_max=args.t_max, T=args.T, t_star=args.t_star
        )
        X_mon = np.vstack([X_ic, X_oc])
        y_mon = np.hstack([y_ic, y_oc])

        X_t = torch.as_tensor(X_mon.astype(np.float32), device=device)
        y_t = torch.as_tensor(y_mon.astype(np.float32), device=device)

        S = score_vectors_last_layer(X_t, y_t, model, args.alpha_lastlayer)
        T2 = ewma_T2(S, args.lam, mu, Sigma_inv)
        exceed += (T2 > UCL).astype(int)

        if (r + 1) % max(1, args.R // 5) == 0:
            print(f"[noise {noise_sd}] PFAR progress {r+1}/{args.R}")

    pfar = exceed / args.R
    return pfar


def plot_pfar(pfar, args, noise_sd, outdir):
    M = args.n_ic + args.n_oc
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(np.arange(M), pfar, lw=2, label=f"Empirical PFAR (noise={noise_sd})")
    ax.axhline(args.alpha_nom, ls="--", color="#d95f02", lw=2, label=f"alpha={args.alpha_nom}")
    ax.axvline(args.n_ic - 0.5, color="k", lw=1.3, ls="--", label="change-point")
    ax.set_xlabel("Sample index i")
    ax.set_ylabel("PFAR (log scale)")
    ax.set_title("Pointwise False Alarm Rate (Zhang et al., example 2)")
    ax.set_ylim(1e-5, 1e0)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"kungang_nonlinear_noise{noise_sd}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"[noise {noise_sd}] wrote {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.03, 0.20])
    ap.add_argument("--n-total", dest="n_total", type=int, default=3000)
    ap.add_argument("--n-ic", dest="n_ic", type=int, default=200)
    ap.add_argument("--n-oc", dest="n_oc", type=int, default=800)
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--lam-cov", dest="lam_cov", type=float, default=1e-1)
    ap.add_argument("--perc", type=float, default=99.9)
    ap.add_argument("--alpha-lastlayer", dest="alpha_lastlayer", type=float, default=1e-1)
    ap.add_argument("--fit-epochs", dest="fit_epochs", type=int, default=3000)
    ap.add_argument("--fit-lr", dest="fit_lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--alpha-nom", dest="alpha_nom", type=float, default=0.001)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--t-max", dest="t_max", type=float, default=30.0)
    ap.add_argument("--t-star", dest="t_star", type=float, default=12.0)
    ap.add_argument("--outdir", default="outputs/kungang_nonlinear")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    for noise_sd in args.noise:
        print(f"[noise {noise_sd}] two-phase fit (n_total={args.n_total}) …")
        t_fit0 = time.perf_counter()
        bundle = fit_two_phase(args, noise_sd, device)
        t_fit_wall = time.perf_counter() - t_fit0
        print(f"[noise {noise_sd}] fit done in {bundle['t_fit']:.1f}s (wall {t_fit_wall:.1f}s), UCL={bundle['UCL']:.3f}")
        t_pfar0 = time.perf_counter()
        pfar = pfar_with_change(bundle, args, noise_sd, device)
        t_pfar_wall = time.perf_counter() - t_pfar0
        base = os.path.join(args.outdir, f"kungang_nonlinear_noise{noise_sd}")
        np.savez(
            base + ".npz",
            pfar=pfar,
            n_ic=args.n_ic,
            n_oc=args.n_oc,
            UCL=bundle["UCL"],
            mu=bundle["mu"],
            Sigma_inv=bundle["Sigma_inv"],
            noise_sd=noise_sd,
            t_fit_wall=t_fit_wall,
            t_fit_inner=bundle["t_fit"],
            t_pfar_wall=t_pfar_wall,
            params=dict(
                n_total=args.n_total,
                R=args.R,
                lam=args.lam,
                lam_cov=args.lam_cov,
                perc=args.perc,
                alpha_lastlayer=args.alpha_lastlayer,
                fit_epochs=args.fit_epochs,
                fit_lr=args.fit_lr,
            ),
        )
        np.savetxt(base + ".csv", np.c_[np.arange(args.n_ic + args.n_oc), pfar], delimiter=",", header="i,FAR", comments="")
        plot_pfar(pfar, args, noise_sd, args.outdir)
        print(f"[noise {noise_sd}] PFAR mean={pfar.mean():.4f}, max={pfar.max():.4f}")


if __name__ == "__main__":
    main()
