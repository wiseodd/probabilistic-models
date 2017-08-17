"""
MLE for Mixture of 1D Linear Regression using EM
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


# Generate data
X1 = np.random.multivariate_normal([1.5, 0.5], np.diag([0.001, 0.001]), size=10)
X2 = np.random.multivariate_normal([0, 0], np.diag([0.01, 0.01]), size=10)
X3 = np.random.multivariate_normal([3, 0], np.diag([0.01, 0.01]), size=10)

X = np.vstack([
    X1[:, 0].reshape(-1, 1), X2[:, 0].reshape(-1, 1), X3[:, 0].reshape(-1, 1)
])
y = np.vstack([
    X1[:, 1].reshape(-1, 1), X2[:, 1].reshape(-1, 1), X3[:, 1].reshape(-1, 1)
])

print(X.shape, y.shape)

N = X.shape[0]
K = 2

# Linear regression params
Ws = np.random.randn(2) * 0.001  # 2x1
beta = np.std(y)  # Global stddev
pis = np.ones(2) * 0.5  # Mixing prior

for it in range(50):
    # E-step
    gammas = np.zeros([N, K])

    for k in range(K):
        lik = st.norm.pdf(X, loc=X*Ws[k], scale=beta)
        gammas[:, k] = (pis[k] * lik).ravel()

    # Evaluate
    loglik = np.sum(np.log(np.sum(gammas, axis=1)))
    print('Iter: {}; loglik: {:.4f}'.format(it, loglik))
    print('W: {}'.format(Ws))
    print('Beta: {}'.format(beta))
    print()

    # Normalize gamma
    gammas = gammas / np.sum(gammas, axis=1)[:, np.newaxis]

    # M-step
    for k in range(K):
        N_k = np.sum(gammas[:, k])
        gamma_k = gammas[:, k]

        # Mixing prob for k-th linreg
        pi = N_k / N
        pis[k] = pi

        # New W for linreg
        R = np.diag(gamma_k)
        W = np.linalg.inv(X.T @ R @ X) @ X.T @ R @ y  # Normal eq.
        Ws[k] = W

    # New beta for linreg
    ssq = np.sum(gammas * (y - X*Ws)**2, axis=1)
    beta = np.sqrt(np.mean(ssq))  # Convert to stddev

# Visualize
xx = np.linspace(-1, 5)

plt.scatter(X, y)
plt.plot(xx, xx*Ws[0])
plt.plot(xx, xx*Ws[1])
plt.show()
