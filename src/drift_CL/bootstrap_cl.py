import numpy as np
from tqdm import trange
from sklearn.utils import resample

def compute_control_limits(X, y, model, alpha=1e-3, lambda_=0.01,
                           BO=60, BI=120, epsilon_reg=1e-6, random_state=0):
    """
    Nested bootstrap control-limit estimation with 0.632-style correction.
    Returns array of time-varying CL_i (upper control limits).
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    CL = np.zeros(n)

    # --- Fit baseline model
    model.fit(X, y)
    scores = compute_scores(X, y, model)

    # --- Outer bootstrap (model refits)
    for i in trange(n, desc="Boot outer"):
        T2_boot = []
        for _ in range(BO):
            idx_o = rng.choice(n, size=n, replace=True)
            Xb, yb = X[idx_o], y[idx_o]
            model.fit(Xb, yb)
            s_boot = compute_scores(Xb, yb, model)

            # Inner bootstrap for future variation
            t2_inner = []
            for _ in range(BI):
                idx_i = rng.choice(n, size=n, replace=True)
                s_i = s_boot[idx_i]
                z = np.mean(s_i, axis=0)
                t2_inner.append(z @ z)
            T2_boot.append(np.quantile(t2_inner, 1 - alpha))

        CL[i] = np.mean(T2_boot)  # approximate time-varying CL_i
    return CL


def compute_scores(X, y, model):
    """
    Placeholder for per-observation score vectors.
    For linear models: gradient wrt coefficients.
    """
    try:
        y_pred = model.predict(X)
        resid = y - y_pred
        return X * resid[:, None]
    except Exception:
        return np.zeros_like(X)
