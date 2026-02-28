#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import socket
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint

# -------------------------
# Simulator / data (trajectory-as-unit; fixed-time response)
# -------------------------
RNG = None


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


# -------------------------
# Your code: model/scores
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--n-tasks", type=int, required=True)

    # experiment params
    ap.add_argument("--B-outer", type=int, default=50)
    ap.add_argument("--B-inner", type=int, default=200)
    ap.add_argument("--n-tr", type=int, default=3000)
    ap.add_argument("--M", type=int, default=1000)
    ap.add_argument("--perc", type=float, default=99.9)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--lam-cov", type=float, default=1e-1)
    ap.add_argument("--alpha-lastlayer", type=float, default=1e-1)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--noise-sd", type=float, default=0.25)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--t-max", dest="t_max", type=float, default=30.0)
    ap.add_argument("--t-star", dest="t_star", type=float, default=12.0)

    ap.add_argument("--fit-epochs-outer", type=int, default=500)
    ap.add_argument("--fit-lr-outer", type=float, default=1e-3)
    ap.add_argument("--fit-epochs-base", type=int, default=3000)
    ap.add_argument("--fit-lr-base", type=float, default=1e-3)
    ap.add_argument(
        "--debug-seed",
        action="store_true",
        help="Log task/boot seeds and a short hash of the first bootstrap draw.",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    wall_start = time.time()

    global RNG
    base_seed = 12345
    seed_task = base_seed + args.task_id
    RNG = np.random.default_rng(np.random.SeedSequence(seed_task))

    # pick device (works even if no GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # baseline params
    base = dict(m1=1.0, m2=2.0, k1=1.0, k2=2.0, k3=1.5, c1=0.1, c2=0.2)

    # generate ONE training set (all tasks should share the same training data)
    # IMPORTANT: for job-array reproducibility, save shared data to disk once.
    shared_path = os.path.join(args.outdir, "shared_train.npz")
    if args.task_id == 0 and not os.path.exists(shared_path):
        X_tr, y_tr = generate_iid_trajectories(
            args.n_tr,
            base,
            noise_sd=args.noise_sd,
            t_max=args.t_max,
            T=args.T,
            t_star=args.t_star,
        )
        X_tr_s = X_tr.astype(np.float32)
        y_tr_s = y_tr.astype(np.float32)
        np.savez(
            shared_path,
            X_tr=X_tr,
            y_tr=y_tr,
            X_tr_s=X_tr_s,
            y_tr_s=y_tr_s,
            scaler_X_mean=np.zeros(X_tr.shape[1], dtype=np.float32),
            scaler_X_scale=np.ones(X_tr.shape[1], dtype=np.float32),
            scaler_y_mean=0.0,
            scaler_y_scale=1.0,
            T=args.T,
            t_max=args.t_max,
            t_star=args.t_star,
        )
        print(f"[task 0] wrote {shared_path}")

    # wait until shared file exists (cheap sync)
    while not os.path.exists(shared_path):
        time.sleep(2)

    dat = np.load(shared_path, allow_pickle=True)
    X_tr_s = dat["X_tr_s"].astype(np.float32)
    y_tr_s = dat["y_tr_s"].astype(np.float32)

    X_tr_t = torch.as_tensor(X_tr_s, device=device)
    y_tr_t = torch.as_tensor(y_tr_s, device=device)

    model_fn = lambda: MLP(X_tr_t.shape[1], hidden=args.hidden, depth=args.depth)

    # baseline model and baseline chart params (done per task; OK, but small overhead)
    t0 = time.perf_counter()
    model0 = train_lastlayer_penalty_only(
        model_fn().to(device),
        X_tr_t,
        y_tr_t,
        alpha_lastlayer=args.alpha_lastlayer,
        epochs=args.fit_epochs_base,
        lr=args.fit_lr_base,
    )
    if args.task_id == 0:
        model_path = os.path.join(args.outdir, "model0_state.pt")
        if not os.path.exists(model_path):
            torch.save(model0.state_dict(), model_path)
            print(f"[task {args.task_id}] wrote {model_path}")
    S_tr = score_vectors_last_layer(X_tr_t, y_tr_t, model0, args.alpha_lastlayer)
    mean_score_norm = float(np.linalg.norm(S_tr.mean(axis=0)))
    print(f"[task {args.task_id}] baseline mean score norm={mean_score_norm:.3e}")
    s_bar = S_tr.mean(axis=0)
    Sigma = np.cov(S_tr.T) + args.lam_cov * np.eye(S_tr.shape[1])
    Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)
    t_base = time.perf_counter() - t0
    print(f"[task {args.task_id}] baseline fit+chart: {t_base:.2f}s on {device}")

    # determine which outer b this task owns
    BO = args.B_outer
    # split BO roughly evenly across tasks
    b_start = (args.task_id * BO) // args.n_tasks
    b_end = ((args.task_id + 1) * BO) // args.n_tasks
    b_list = list(range(b_start, b_end))
    BO_chunk = len(b_list)
    print(f"[task {args.task_id}] handling outer b in [{b_start}, {b_end}) = {BO_chunk} reps")

    # prealloc
    res = np.empty((BO_chunk * args.B_inner, args.M), dtype=float)
    t_fit = np.zeros(BO_chunk)
    t_score = np.zeros(BO_chunk)
    t_inner = np.zeros(BO_chunk)
    t_total = np.zeros(BO_chunk)

    n_tr = args.n_tr
    idx_all = np.arange(n_tr)

    # main chunk loop
    for ii, b in enumerate(b_list):
        tB0 = time.perf_counter()

        seed_b = base_seed + (b + 1) * 100000 + args.task_id
        rng_b = np.random.default_rng(np.random.SeedSequence(seed_b))
        idx_boot = rng_b.choice(n_tr, n_tr, replace=True)
        if args.debug_seed and b in (0, 1):
            h = hashlib.sha256(idx_boot[:20].tobytes()).hexdigest()[:12]
            print(
                f"[task {args.task_id}] seed_task={seed_task} seed_b={seed_b} "
                f"idx_boot_hash={h}"
            )
        mask = np.ones(n_tr, dtype=bool)
        mask[np.unique(idx_boot)] = False
        idx_oob = idx_all[mask]

        # fit
        t1 = time.perf_counter()
        mdl_b = train_lastlayer_penalty_only(
            model_fn().to(device),
            X_tr_t[idx_boot],
            y_tr_t[idx_boot],
            alpha_lastlayer=args.alpha_lastlayer,
            epochs=args.fit_epochs_outer,
            lr=args.fit_lr_outer,
        )
        t2 = time.perf_counter()
        t_fit[ii] = t2 - t1

        # scores
        t3 = time.perf_counter()
        S_b = score_vectors_last_layer(
            X_tr_t[idx_boot], y_tr_t[idx_boot], mdl_b, args.alpha_lastlayer
        )
        if ii == 0:
            mean_score_norm_b = float(np.linalg.norm(S_b.mean(axis=0)))
            print(f"[task {args.task_id}] bootstrap b={b} mean score norm={mean_score_norm_b:.3e}")
        s_bar_b = S_b.mean(axis=0)
        Sigma_b = np.cov(S_b.T) + args.lam_cov * np.eye(S_b.shape[1])
        Sigma_b_inv = np.linalg.pinv(Sigma_b, hermitian=True)

        if len(idx_oob) == 0:
            S_oob = S_b
        else:
            S_oob = score_vectors_last_layer(
                X_tr_t[idx_oob], y_tr_t[idx_oob], mdl_b, args.alpha_lastlayer
            )
        t4 = time.perf_counter()
        t_score[ii] = t4 - t3

        # inner
        t5 = time.perf_counter()
        for j in range(args.B_inner):
            idx_inner = rng_b.choice(len(S_oob), args.M, replace=True)
            stream = S_oob[idx_inner]
            res[ii * args.B_inner + j] = ewma_T2(
                stream, args.lam, s_bar_b, Sigma_b_inv, n_train=n_tr, apply_k_correction=True
            )
        t6 = time.perf_counter()
        t_inner[ii] = t6 - t5

        t_total[ii] = time.perf_counter() - tB0

        if (ii + 1) % max(1, BO_chunk // 5) == 0:
            print(
                f"[task {args.task_id}] progress {ii+1}/{BO_chunk}, "
                f"median total={np.median(t_total[:ii+1]):.2f}s"
            )

    # save
    outpath = os.path.join(args.outdir, f"chunk_task{args.task_id:03d}.npz")
    meta = dict(
        task_id=args.task_id,
        n_tasks=args.n_tasks,
        b_start=b_start,
        b_end=b_end,
        B_outer=args.B_outer,
        B_inner=args.B_inner,
        n_tr=args.n_tr,
        M=args.M,
        perc=args.perc,
        lam=args.lam,
        lam_cov=args.lam_cov,
        alpha_lastlayer=args.alpha_lastlayer,
        hidden=args.hidden,
        depth=args.depth,
        noise_sd=args.noise_sd,
        T=args.T,
        t_max=args.t_max,
        t_star=args.t_star,
        device=str(device),
        scaler_X_mean=np.zeros(X_tr_s.shape[1], dtype=float).tolist(),
        scaler_X_scale=np.ones(X_tr_s.shape[1], dtype=float).tolist(),
        scaler_y_mean=0.0,
        scaler_y_scale=1.0,
        baseline_time=t_base,
    )
    np.savez(
        outpath,
        res=res,
        t_fit=t_fit,
        t_score=t_score,
        t_inner=t_inner,
        t_total=t_total,
        s_bar=s_bar,
        Sigma_inv=Sigma_inv,
        meta=json.dumps(meta),
    )
    print(f"[task {args.task_id}] wrote {outpath}")

    # write wall-clock timing metadata for parallel aggregation
    wall_end = time.time()
    clock = dict(
        task_id=args.task_id,
        n_tasks=args.n_tasks,
        b_start=b_start,
        b_end=b_end,
        B_outer=args.B_outer,
        B_inner=args.B_inner,
        n_tr=args.n_tr,
        M=args.M,
        noise_sd=args.noise_sd,
        wall_start=wall_start,
        wall_end=wall_end,
        wall_seconds=wall_end - wall_start,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        device=str(device),
    )
    clock_path = os.path.join(args.outdir, f"clock_task{args.task_id:03d}.json")
    with open(clock_path, "w", encoding="utf-8") as f:
        json.dump(clock, f)
    print(f"[task {args.task_id}] wrote {clock_path}")


if __name__ == "__main__":
    main()
