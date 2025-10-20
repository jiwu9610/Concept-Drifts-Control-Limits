"""
Runs the linear mixture example and reproduces the main figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from driftcl.bootstrap_cl import compute_control_limits
from driftcl.monitoring import monitor_stream

# Load or generate data (placeholder)
X = np.random.randn(2000, 5)
y = X @ np.array([1, -1, 0.5, 0, 0.2]) + np.random.randn(2000) * 0.1
model = Ridge(alpha=0.1)

# Split training/stream
X_train, y_train = X[:1000], y[:1000]
X_stream, y_stream = X[1000:], y[1000:]

# Compute CLs and monitor
CL = compute_control_limits(X_train, y_train, model)
results = monitor_stream(X_stream, y_stream, model, CL)

# Plot
plt.figure(figsize=(6,3))
plt.plot(results["T2"], label="T² statistic")
plt.plot(CL[:len(results["T2"])], label="UCL")
if results["signal_index"]:
    plt.axvline(results["signal_index"], color="r", linestyle="--", label="First signal")
plt.legend()
plt.xlabel("Observation index")
plt.ylabel("Statistic")
plt.tight_layout()
plt.savefig("outputs/mixture/t2_vs_ucl.png", dpi=200)
print("Figure saved to outputs/mixture/t2_vs_ucl.png")
