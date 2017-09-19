"""
Stick Breaking Construction
---------------------------
Generative story of Stick Breaking process
"""
import numpy as np


alpha = 1  # dirichlet prior
n_clusters = 10  # max num of clusters

G = lambda: np.random.normal()  # base distribution, theta ~ G

thetas = np.zeros(n_clusters)  # cluster params
phis = np.zeros(n_clusters)  # cluster mixing weights

prev_v = 1

for i in range(n_clusters):
    # v ~ Beta(1, alpha)
    v = np.random.beta(1, alpha)
    # theta ~ G
    theta = G()

    thetas[i] = theta
    phis[i] = v * prev_v  # v_1 * \prod_{i>1} v_i

    prev_v *= (1-v)

# Check: phis should sum to one as n_clusters -> inf
print(np.sum(phis))
