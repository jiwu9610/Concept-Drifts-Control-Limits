import numpy as np

def regularize_cov(Sigma, eps=1e-6):
    """
    Adds eps * I for numerical stability.
    """
    return Sigma + eps * np.eye(Sigma.shape[0])
