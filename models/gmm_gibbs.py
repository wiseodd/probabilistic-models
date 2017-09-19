"""
Posterior sampling for Gaussian Mixture Model using Gibbs sampler
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
lambdas = np.array([np.linalg.inv(sigmas[0]), np.linalg.inv(sigmas[1])])
pis = np.array([0.5, 0.5])  # Mixing probs.
zs = np.zeros([N])  # Assignments

# Priors
alpha = np.array([1, 1])
pis = np.random.dirichlet(alpha)
mus0 = np.array([[1, 1], [1, 1]], dtype='float')
sigmas0 = np.array([np.diag([1, 1]), np.diag([1, 1])], dtype='float')
lambdas0 = np.array([np.linalg.inv(sigmas0[0]), np.linalg.inv(sigmas0[1])])

# Gibbs sampler
for it in range(50):
    # Sample from full conditional of assignment
    # z ~ p(z) \propto pi*N(y|pi)
    probs = np.zeros([N, K])

    for k in range(K):
        p = pis[k] * st.multivariate_normal.pdf(X, mean=mus[k], cov=sigmas[k])
        probs[:, k] = p

    # Normalize
    probs /= np.sum(probs, axis=1)[:, np.newaxis]

    # For each data point, draw the cluster assignment
    for i in range(N):
        z = np.random.multinomial(n=1, pvals=probs[i]).argmax()
        zs[i] = z

    # Sample from full conditional of cluster parameter
    # Assume fixed covariance => posterior is Normal
    # mu ~ N(mu, sigma)
    Ns = np.zeros(K, dtype='int')

    for k in range(K):
        # Gather all data points assigned to cluster k
        Xk = X[zs == k]
        Ns[k] = Xk.shape[0]

        # Covariance of posterior
        lambda_post = lambdas0[k] + Ns[k]*lambdas[k]
        cov_post = np.linalg.inv(lambda_post)

        # Mean of posterior
        left = cov_post
        right = lambdas0[k] @ mus0[k] + Ns[k]*lambdas[k] @ np.mean(Xk, axis=0)
        mus_post = left @ right

        # Draw new mean sample from posterior
        mus[k] = st.multivariate_normal.rvs(mus_post, cov_post)

    # Sample from full conditional of the mixing weight
    # pi ~ Dir(alpha + n)
    pis = np.random.dirichlet(alpha + Ns)

# Expected output:
# ----------------
# 20 data in cluster-0, mean: [ 5  5 ]
# 20 data in cluster-1, mean: [ 8  8 ]
for k in range(K):
    print('{} data in cluster-{}, mean: {}'.format(Ns[k], k, mus[k]))
