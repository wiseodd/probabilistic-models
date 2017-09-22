"""
LDA with Collapsed Gibbs Sampler
================================
Reference: Kevin Murphy's book Ch. 27
"""
import numpy as np


# Words
W = np.array([0, 1, 2, 3, 4])

# X := document words
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

# Dirichlet priors
alpha = 1
gamma = 1


# --------------
# Initialization
# --------------

# Z := word topic assignment
Z = np.zeros(shape=[N_D, N_W], dtype=int)

# Some counts for sufficient statistics
c_ik = np.zeros(shape=[N_D, N_K], dtype=int)
c_vk = np.zeros(shape=[N_W, N_K], dtype=int)
c_k = np.zeros(shape=N_K, dtype=int)

for i in range(N_D):
    for v in range(N_W):
        # Randomly assign word's topic
        k = np.random.randint(N_K)
        Z[i, v] = k

        # Record counts
        c_ik[i, k] += 1
        c_vk[v, k] += 1
        c_k[k] += 1


L = np.array([x.size for x in X])


# --------------
# Gibbs sampling
# --------------

for it in range(1000):
    # Sample from full conditional of Z
    # ---------------------------------
    for i in range(N_D):
        for v in range(N_W):
            # Sufficient statistics for the full conditional
            k = Z[i, v]
            c_ik[i, k] -= 1
            c_vk[v, k] -= 1
            c_k[k] -= 1

            # Calculate full conditional p(z_iv | .)
            left = (c_vk[v, :] + gamma) / (c_k + N_W*gamma)
            right = (c_ik[i, :] + alpha) / (L[i] + N_K*alpha)

            p_z_iv = left * right
            p_z_iv /= np.sum(p_z_iv)

            # Resample word topic assignment
            k = np.random.multinomial(1, p_z_iv).argmax()

            # Update counts
            Z[i, v] = k
            c_ik[i, k] += 1
            c_vk[v, k] += 1
            c_k[k] += 1


print('Documents:')
print('----------')
print(X)

print()

print('Document topic distribution:')
print('----------------------------')
print((c_ik + alpha) / np.sum(c_ik + alpha, axis=1)[:, np.newaxis])

print()

print('Topic word distribution:')
print('----------------------------')
x = (c_vk + gamma).T
print(x / np.sum(x, axis=1)[:, np.newaxis])
