"""
Posterior sampling for Gaussian Mixture Model using Gibbs sampler
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


K = 2

# Generate data
X1 = np.random.normal(-5, 0.5, size=20)
X2 = np.random.normal(3, 0.5, size=20)
X = np.concatenate([X1, X2])

N = X.shape[0]

# GMM params initialization
mu = np.array([50, -20], dtype=float)
var = np.array([1, 1], dtype=float)
phi = np.zeros([N, K]) + 1/K
c = np.zeros([N, K])  # Assignments

# Priors
mu0 = 0
var0 = 1

for it in range(5):
    # Update variational param phi, the assignment probs
    for k in range(K):
        phi[:, k] = np.exp(mu[k]*X - (mu[k]**2 + var[k])/2)

    # Normalize
    phi /= np.sum(phi, axis=1)[:, np.newaxis]
    # Update assignments
    c[np.argmax(phi, axis=1)] = 1

    # Update variational param mu and var, the params of Gaussian component
    for k in range(K):
        sum_phi = np.sum(phi[:, k])
        mu[k] = phi[:, k] @ X / (1/var0 + sum_phi)
        var[k] = 1 / (1/var0 + sum_phi)


# Expected output:
# ----------------
# 20 data in cluster-0, mean: -5
# 20 data in cluster-1, mean: 3
for k in range(K):
    n = np.sum(c == k)
    print('{} data in cluster-{}, mean: {}'.format(n, k, mu[k]))