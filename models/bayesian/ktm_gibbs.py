"""
Kernel topic model with Gibbs Sampler
=====================================
Reference: Hennig et al., 2012 & Murphy's MLPP book Ch. 27
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm, trange


np.random.seed(1)

# Words
W = np.array([0, 1, 2, 3, 4])

# D := document words
X = np.array([
    [0, 0, 1, 2, 2],
    [0, 0, 1, 1, 1],
    [0, 1, 2, 2, 2],
    [4, 4, 4, 4, 4],
    [3, 3, 4, 4, 4],
    [3, 4, 4, 4, 4]
])

N_D = X.shape[0]  # num of docs
N_W = W.shape[0]  # num of words
N_K = 2  # num of topics
N_F = 3  # num of features

# Document features
Phi = np.random.randn(N_D, N_F)

# Dirichlet priors
alpha = 1
beta = 1

# k independent GP priors
ls = 1  # length-scale for RBF kernel
tau = 1  # Observation noise variance
kernel = RBF([ls]*N_F)
GPRs = []
for k in range(N_K):
    GPR_k = GPR(kernel=kernel, alpha=tau)
    GPR_k = GPR_k.fit(Phi, np.zeros(N_D))
    GPRs.append(GPR_k)


# -------------------------------------------------------------------------------------
# Laplace bridge
# -------------------------------------------------------------------------------------

def gauss2dir(mu, Sigma):
    K = len(mu)
    Sigma_diag = np.diag(Sigma)

    alpha = 1/Sigma_diag * (1 - 2/K + np.exp(mu)/K**2 * np.sum(np.exp(-mu)))

    return alpha


def dir2gauss(alpha):
    K = len(alpha)

    mu = np.log(alpha) - 1/K*np.sum(np.log(alpha))
    Sigma = np.diag(1/alpha)

    for k in range(K):
        for l in range(K):
            Sigma[k, l] -= 1/K*(1/alpha[k] + 1/alpha[l] - 1/K*np.sum(1/alpha))

    return mu, Sigma


# -------------------------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------------------------

# Pi := document topic distribution
Pi = np.zeros([N_D, N_K])
for d in range(N_D):
    Pi[d] = np.random.dirichlet(alpha*np.ones(N_K))

# C := word topic assignment
C = np.zeros(shape=[N_D, N_W])
for d in range(N_D):
    for i in range(N_W):
        C[d, i] = np.random.multinomial(1, Pi[d]).argmax()

# Theta := topic descriptions
Theta = np.zeros([N_K, N_W])
for k in range(N_K):
    Theta[k] = np.random.dirichlet(beta*np.ones(N_W))


# -------------------------------------------------------------------------------------
# Inference of Pi, Theta, and C with (non-collapsed) Gibbs sampling
# -------------------------------------------------------------------------------------

for it in trange(100):
    # Sample Pi
    # ----------------------------------

    for d in range(N_D):

        mu_y_d = np.zeros(N_K)

        for k in range(N_K):
            mu_y_d[k] = GPRs[k].predict(Phi[d].reshape(-1, N_F))

        # Use the Laplace bridge to get a Dirichlet belief
        param_pi_d = gauss2dir(mu_y_d, tau*np.eye(N_K))

        # Gather counts
        nu_d = np.zeros(N_K)
        for k in range(N_K):
            nu_d[k] = np.sum(C[d] == k)

        # Sample from the full conditional of pi
        Pi[d, :] = np.random.dirichlet(param_pi_d + nu_d)


    # Sample Theta
    # ---------------------------------

    for k in range(N_K):
        eta = np.zeros(N_W)

        # Gather counts
        for j in range(N_W):
            for d in range(N_D):
                for i in range(N_W):
                    eta[j] += (X[d, i] == j) and (C[d, i] == k)

        # Resample word topic dist.
        Theta[k, :] = np.random.dirichlet(beta + eta)


    # Sample C
    # ---------------------------------

    for d in range(N_D):
        for i in range(N_W):
            # Full-conditional of c_di
            p_bar_di = np.exp(np.log(Pi[d]) + np.log(Theta[:, X[d, i]]))
            p_di = p_bar_di / np.sum(p_bar_di)

            # Sample C from the full-conditional
            C[d, i] = np.random.multinomial(1, p_di).argmax()


    # Update the GPs
    # ---------------------------------

    # Get P(y_d) from the updated P(pi_d) using the inverse Laplace bridge
    Y = np.zeros([N_D, N_K])
    for d in range(N_D):
        mean, cov = dir2gauss(Pi[d])
        Y[d, :] = mean

    for k in range(N_K):
        GPRs[k].fit(Phi, Y[:, k])


print('Documents:')
print('----------')
print(X)

print()

print('Document topic distribution:')
print('----------------------------')
print(Pi)

print()

print('Topic\'s word distribution:')
print('-------------------------')
print(Theta)

print()

print('Word topic assignment:')
print('-------------------------')
print(C)
