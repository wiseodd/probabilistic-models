"""
Chinese Restaurant Process
--------------------------
Generative story of CRP
"""
import numpy as np


N = 10  # num of customer
alpha = 1  # dirichlet prior

C = np.zeros(N)  # cust. table assignment
K = 0  # num of current table
tables = []  # cust. for tables

for n in range(N):
    # Join a table with prob propto # of people
    p_join = np.zeros(K)

    for k in range(K):
        p_join[k] = tables[k] / (n + alpha)

    # Open new table with prob propto alpha
    p_new = alpha / (n + alpha)

    # Draw table n-th cust will join
    p = np.append(p_join, p_new)
    t = np.random.multinomial(n=1, pvals=p).argmax()

    # Update table count
    if t >= K:
        K += 1
        tables.append(0)

    C[n] = t  # assign cust n to table t
    tables[t] += 1  # increment cust count for t-th table

print(C)
