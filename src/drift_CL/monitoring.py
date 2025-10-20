import numpy as np
from .mewma import update_mewma, compute_t2

def monitor_stream(X_stream, y_stream, model, CL, lambda_=0.01):
    """
    Stream monitoring: compute MEWMA statistics and compare to CL_i.
    """
    n = len(y_stream)
    s_bar = np.zeros(X_stream.shape[1])
    sigma_inv = np.eye(X_stream.shape[1])
    z = np.zeros_like(s_bar)

    T2, signal_index = [], None
    for i in range(n):
        s_i = X_stream[i] * (y_stream[i] - model.predict(X_stream[i].reshape(1, -1)))
        z = update_mewma(z, s_i, lambda_)
        t2 = compute_t2(z, s_bar, sigma_inv)
        T2.append(t2)
        if signal_index is None and t2 > CL[min(i, len(CL)-1)]:
            signal_index = i
    return {"T2": np.array(T2), "signal_index": signal_index}
