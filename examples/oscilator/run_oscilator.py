"""
Reproduces the nonlinear oscillator example and saves a T² vs UCL plot.

This uses a simple nonlinear oscillator simulator to generate a training
segment (pre-shift) and a streaming segment (post-shift). We fit a small
neural net (scikit-learn MLP) to the pre-shift segment and monitor the stream
with bootstrap-calibrated, time-varying control limits.

Outputs:
  - outputs/oscillator/t2_vs_ucl.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.drift_CL.bootstrap_cl import compute_control_limits
from src.drift_CL.monitoring import monitor_stream

# -----------------------
# 1) Nonlinear oscillator
# -----------------------
def simulate_oscillator(n_steps=3000, dt=0.02, omega=1.6, damping=0.05,
                        noise_std=0.01, force_shift_at=None, force_scale=1.0, seed=0):
    """
    State: [x, v]
    x_{t+1} = x_t + v_t * dt
    v_{t+1} = v_t - (omega^2) * sin(x_t) * dt - damping * v_t * dt + epsilon_t
    Optional: apply a multiplicative change in omega after `force_shift_at`.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)

    x[0] = 0.5
    v[0] = 0.0

    for t in range(n_steps - 1):
        om = omega if (force_shift_at is None or t < force_shift_at) else omega * force_scale
        a = - (om ** 2) * np.sin(x[t]) - damping * v[t]
        v[t + 1] = v[t] + a * dt + rng.normal(0.0, noise_std)
        x[t + 1] = x[t] + v[t] * dt

    return x, v

def make_supervised(x, v, lag=3):
    """
    Build supervised features from lagged states.
    Predict next x(t+1) from [x(t), v(t), x(t-1), v(t-1), ...].
    """
    T = len(x)
    feats, target = [], []
    for t in range(lag, T - 1):
        row = []
        for k in range(lag):
            row += [x[t - k], v[t - k]]
        feats.append(row)
        target.append(x[t + 1])
    return np.asarray(feats), np.asarray(target)

# -----------------------
# 2) Generate data
# -----------------------
os.makedirs("outputs/oscillator", exist_ok=True)

# Pre-shift + post-shift stream with a parameter change at t_shift
t_shift = 1500
x, v = simulate_oscillator(n_steps=3200, force_shift_at=t_shift, force_scale=1.25, seed=42)

# Train on the first 1400 (entirely pre-shift), stream on the remainder
lag = 4
X_all, y_all = make_supervised(x, v, lag=lag)

# Map indices from x/v to supervised indices:
# supervised index t corresponds roughly to original index t+1
train_end_sup = 1350                      # strictly pre-shift
X_train, y_train = X_all[:train_end_sup], y_all[:train_end_sup]
X_stream, y_stream = X_all[train_end_sup:], y_all[train_end_sup:]

# -----------------------
# 3) Fit base model (small NN)
# -----------------------
# Use sklearn MLP to keep `predict` API consistent with driftcl stubs
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(32,),
                         activation="relu",
                         alpha=1e-3,
                         max_iter=300,
                         random_state=0))
])
model.fit(X_train, y_train)

# -----------------------
# 4) Compute control limits + monitor
# -----------------------
CL = compute_control_limits(
    X_train, y_train, model=model,
    alpha=1e-3, lambda_=0.01,
    BO=60, BI=120, epsilon_reg=1e-6, random_state=0
)

results = monitor_stream(X_stream, y_stream, model, CL, lambda_=0.01)

# -----------------------
# 5) Plot: T² vs UCL
# -----------------------
plt.figure(figsize=(7.0, 3.4))
plt.plot(results["T2"], label="T² statistic")
plt.plot(CL[:len(results["T2"])], label="UCL")

if results["signal_index"] is not None:
    plt.axvline(results["signal_index"], linestyle="--", label="First signal")

plt.title("Nonlinear Oscillator: T² vs. UCL")
plt.xlabel("Stream index")
plt.ylabel("Statistic")
plt.legend()
plt.tight_layout()
out_path = "outputs/oscillator/t2_vs_ucl.png"
plt.savefig(out_path, dpi=200)
print(f"Figure saved to {out_path}")
