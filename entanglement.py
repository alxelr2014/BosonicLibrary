import numpy as np


def schmidt_coeffs(rho):
    u,s,vh = np.linalg.svd(rho,full_matrices=True)
    return s


def f(x):
    if x <= 1e-5:
        return 0
    return -x*np.log2(x)

def entropy(probs):
    return sum(np.vectorize(f)(probs))
