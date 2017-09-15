"""
Indian Buffet Process
---------------------
Generative story for IBP
"""
import numpy as np


N = 10  # num of customer

C = [[] for i in range(N)]  # cust. dishes assignment
K = 0  # num of current dishes selected
dishes = []  # num. of cust. sampled each dish
alpha = 1  # Poisson prior

for n in range(N):
    # Sample each dish, w.p. propto the popularity of dishes
    for k in range(K):
        p = dishes[k] / n
        z = np.random.binomial(n=1, p=p)
        C[n].append(z)  # update cust. dishes assignment
        dishes[k] += 1  # update cust. count for dish k

    # Try out new dishes acc. to Poi(alpha/n)
    new_dishes = np.random.poisson(lam=alpha)

    for k in range(new_dishes):
        C[n].append(1)
        dishes.append(1)

    # Update dish count
    K += new_dishes

# Convert to binary matrix
C_ = np.zeros([N, K])

for n in range(N):
    C_[n, 0:len(C[n])] = np.array(C[n])

print(C_)
