"""
LDA with Gibbs Sampler
======================
Reference: Blei DM, Kucukelbir A, McAuliffe JD. Variational inference: A review for statisticians. Journal of the American Statistical Association. 2017.
"""
import numpy as np
from scipy.special import digamma


# Words
V = np.array([0, 1, 2, 3, 4])

# D := document words
X = np.array([
    [0, 0, 1, 2, 2],
    [0, 0, 1, 1, 2],
    [0, 1, 2, 2, 2],
    [4, 4, 4, 4, 4],
    [3, 3, 4, 4, 4],
    [3, 4, 4, 4, 4]
])

N_D = X.shape[0]  # num of docs
N_V = V.shape[0]  # num of words
N_K = 2  # num of topics

W = np.zeros([N_D, N_V, N_V])
for d in range(N_D):
    for n in range(N_V):
        W[d, n, X[d, n]] = 1


# Dirichlet priors
alpha = 1
eta = 1


# --------------
# Initialization
# --------------

phi = np.random.randn(N_D, N_V, N_K)
phi[d, n, :] /= np.sum(phi[d, n, :])  # normalize

gamma = np.random.randn(N_D, N_K)
gamma /= np.sum(gamma, axis=1)[:, np.newaxis]  # normalize

lam = np.random.randn(N_K, N_V)
lam /= np.sum(lam, axis=1)[:, np.newaxis]  # normalize


# -------------
# Mean-field VI
# -------------

for it in range(100):
    # Until phi and gamma converge:
    for _ in range(50):
        for d in range(N_D):
            for n in range(N_V):
                left = digamma(gamma[d, :]) + digamma(lam[:, X[d, n]])
                right = digamma(np.sum(lam, axis=1))
                phi[d, n, :] = np.exp(left - right)
                phi[d, n, :] /= np.sum(phi[d, n, :])  # normalize

        gamma = alpha + np.sum(phi, axis=1)
        gamma /= np.sum(gamma, axis=1)[:, np.newaxis]  # normalize

    # Update lam given phi and gamma:
    for k in range(N_K):
        lam[k] = eta  # prior

        for d in range(N_D):
            for n in range(N_V):
                # Count words weighted by word-topic distribution
                lam[k] += phi[d, n, k] * W[d, n]

    lam /= np.sum(lam, axis=1)[:, np.newaxis]  # normalize


# -------
# Results
# -------

print('Documents:')
print('----------')
print(X)

print()

print('Document topic proportion:\n{}'.format(gamma), end='\n\n')
print('Topic word proportion:\n{}'.format(lam))
