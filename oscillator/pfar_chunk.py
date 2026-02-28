#!/usr/bin/env python3
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler

RNG = None


# --- simulator / data (trajectory-as-unit; fixed-time response) ---
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


def sample_one_trajectory_fixed_time(
    params, noise_sd, *, t_max=30.0, T=20, t_star=12.0, init_sampler=None
):
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


def generate_iid_trajectories(
    n_samples, params, *, noise_sd=0.03, t_max=30.0, T=20, t_star=12.0, init_sampler=None
):
    X = np.empty((n_samples, 4 * T), dtype=float)
    y = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        Xi, yi = sample_one_trajectory_fixed_time(
            params,
            noise_sd,
            t_max=t_max,
            T=T,
            t_star=t_star,
            init_sampler=init_sampler,
        )
        X[i] = Xi
        y[i] = yi
    return X, y


# --- model / scores ---
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


# --- EWMA T2 ---
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--n-tasks", type=int, required=True)
    ap.add_argument("--R", type=int, default=200)  # total MC runs across all tasks

    # need these consistent with bootstrap run
    ap.add_argument("--n-tr", type=int, default=3000)
    ap.add_argument("--M", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--alpha-lastlayer", type=float, default=1e-3)
    ap.add_argument("--lam-cov", dest="lam_cov", type=float, default=1e-1)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--noise-sd", type=float, default=0.25)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--t-max", dest="t_max", type=float, default=30.0)
    ap.add_argument("--t-star", dest="t_star", type=float, default=12.0)
    ap.add_argument("--fit-epochs-base", type=int, default=3000)
    ap.add_argument("--fit-lr-base", type=float, default=1e-3)
    args = ap.parse_args()

    global RNG
    base_seed = 23456
    RNG = np.random.default_rng(np.random.SeedSequence(base_seed + args.task_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shared_path = os.path.join(args.outdir, "shared_train.npz")
    ucl_path = os.path.join(args.outdir, "ucl_and_chartparams.npz")
    assert os.path.exists(shared_path), "shared_train.npz missing"
    assert os.path.exists(ucl_path), "ucl_and_chartparams.npz missing"

    dat = np.load(shared_path, allow_pickle=True)
    X_tr_s = dat["X_tr_s"].astype(np.float32)
    y_tr_s = dat["y_tr_s"].astype(np.float32)

    # no scaling; keep identity transforms for compatibility
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.mean_ = np.zeros(X_tr_s.shape[1], dtype=float)
    scaler_X.scale_ = np.ones(X_tr_s.shape[1], dtype=float)
    scaler_X.var_ = np.ones(X_tr_s.shape[1], dtype=float)
    scaler_X.n_features_in_ = X_tr_s.shape[1]
    scaler_y.mean_ = np.array([0.0])
    scaler_y.scale_ = np.array([1.0])
    scaler_y.var_ = np.array([1.0])
    scaler_y.n_features_in_ = 1

    chart = np.load(ucl_path, allow_pickle=True)
    UCL = chart["UCL"]
    if "s_bar" not in chart or "Sigma_inv" not in chart:
        raise KeyError("ucl_and_chartparams.npz missing s_bar or Sigma_inv")
    s_bar = chart["s_bar"]
    Sigma_inv = chart["Sigma_inv"]
    meta = {}
    if "meta" in chart:
        try:
            meta = json.loads(str(chart["meta"]))
        except Exception:
            meta = {}
    t_max = float(meta.get("t_max", args.t_max))
    T = int(meta.get("T", args.T))
    t_star = float(meta.get("t_star", args.t_star))

    X_tr_t = torch.as_tensor(X_tr_s, device=device)
    y_tr_t = torch.as_tensor(y_tr_s, device=device)
    model_fn = lambda: MLP(X_tr_t.shape[1], hidden=args.hidden, depth=args.depth)

    # require baseline model from bootstrap run to keep PFAR aligned to UCL
    model0 = model_fn().to(device)
    model_path = os.path.join(args.outdir, "model0_state.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "model0_state.pt missing; rerun bootstrap (task 0) before PFAR."
        )
    model0.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[task {args.task_id}] loaded baseline model from {model_path}")
    mean_score_norm = float(np.linalg.norm(s_bar))
    print(f"[task {args.task_id}] baseline mean score norm (from UCL)={mean_score_norm:.3e}")

    base = dict(m1=1.0, m2=2.0, k1=1.0, k2=2.0, k3=1.5, c1=0.1, c2=0.2)

    # split R across tasks
    R_total = args.R
    r_start = (args.task_id * R_total) // args.n_tasks
    r_end = ((args.task_id + 1) * R_total) // args.n_tasks
    R_chunk = r_end - r_start
    print(f"[task {args.task_id}] PFAR runs {r_start}..{r_end-1} (R_chunk={R_chunk}) on {device}")

    exceed = np.zeros(args.M, dtype=int)

    for _ in range(R_chunk):
        X_ic, y_ic = generate_iid_trajectories(
            args.M,
            base,
            noise_sd=args.noise_sd,
            t_max=t_max,
            T=T,
            t_star=t_star,
        )
        X_ic_s = scaler_X.transform(X_ic).astype(np.float32)
        y_ic_s = scaler_y.transform(y_ic[:, None]).ravel().astype(np.float32)

        X_t = torch.as_tensor(X_ic_s, device=device)
        y_t = torch.as_tensor(y_ic_s, device=device)

        S = score_vectors_last_layer(X_t, y_t, model0, args.alpha_lastlayer)
        T2 = ewma_T2(S, args.lam, s_bar, Sigma_inv, n_train=args.n_tr, apply_k_correction=False)
        exceed += (T2 > UCL).astype(int)

    outpath = os.path.join(args.outdir, f"pfar_task{args.task_id:03d}.npz")
    np.savez(outpath, exceed=exceed, R_chunk=R_chunk)
    print(f"[task {args.task_id}] wrote {outpath}")


if __name__ == "__main__":
    main()
