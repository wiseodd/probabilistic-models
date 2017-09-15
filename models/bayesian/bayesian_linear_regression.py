"""
Bayesian Linear Regression
-----------------
Bayesian Linear Regression with toy data in 1D.
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


# Generate data
M = 20

X = np.linspace(0, 2, num=M)
# X = np.array([5, 14, 19], dtype=np.float)
t_real = np.sin(X)
t = t_real + np.random.randn(M) * 0.25

plt.scatter(X, t, label='Data points')

# Generative process: p(t|W,X,beta) = N(t|XW+b,beta)
beta = 1

# Prior: N(w|0,1/alpha*I)
alpha = 1

# Posterior: N(w|m,s):
s = 1/(alpha + beta * X.T @ X)
m = beta * s * X.T @ t

# Infer p(t|t,alpha,beta) the predictive distribution
X_pred = np.linspace(0, 2, num=100)

m_pred = m * X_pred
s_pred = 1/beta + X_pred.T @ X_pred * s
std_pred = np.sqrt(s_pred)

plt.plot(X_pred, m_pred, color='red', alpha=0.75, label='Regression line')
plt.fill_between(
    X_pred, m_pred-std_pred, m_pred+std_pred,
    interpolate=True, color='green', alpha=0.1, label='+- 1 stddev'
)

# Sample from predictive dist.
ys = np.random.normal(m_pred, std_pred)

plt.plot(X_pred, ys, alpha=0.15, label='Posterior samples')
plt.legend(loc='best')
plt.show()
