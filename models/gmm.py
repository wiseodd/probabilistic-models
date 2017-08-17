"""
MLE for Gaussian Mixture Model using EM
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


K = 2

# Generate data
X1 = np.random.multivariate_normal([5, 5], np.diag([0.5, 0.5]), size=20)
X2 = np.random.multivariate_normal([8, 8], np.diag([0.5, 0.5]), size=20)
X = np.vstack([X1, X2])

N = X.shape[0]

# GMM params
mus = np.array([[1, 1], [15, 15]], dtype='float')
sigmas = np.array([np.diag([1, 1]), np.diag([1, 1])], dtype='float')
pis = np.array([0.5, 0.5])

for it in range(5):
    # E-step
    gammas = np.zeros([N, K])

    for k in range(K):
        lik = st.multivariate_normal.pdf(X, mean=mus[k], cov=sigmas[k])
        gammas[:, k] = pis[k] * lik

    # Evaluate
    loglik = np.sum(np.log(np.sum(gammas, axis=1)))
    print('Log-likelihood: {:.4f}'.format(loglik))
    print('Mus: {}'.format(mus))
    print()

    # Normalize gamma
    gammas = gammas / np.sum(gammas, axis=1)[:, np.newaxis]

    # M-step
    for k in range(K):
        Nk = np.sum(gammas[:, k])

        mu = 1/Nk * np.sum(gammas[:, k][:, np.newaxis] * X, axis=0)

        Xmu = (X - mu)[:, :, np.newaxis]
        sigma = 1/Nk * np.sum(
            [gammas[i, k] * Xmu[i] @ Xmu[i].T for i in range(N)],
            axis=0
        )

        pi = Nk / N

        mus[k] = mu
        sigmas[k] = sigma
        pis[k] = pi
