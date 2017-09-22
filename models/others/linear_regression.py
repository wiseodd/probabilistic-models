"""
Linear Regression
-----------------
Probabilistic Linear Regression with toy data in 1D.
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

# Infer p(t|W,X,alpha) = N(t|XW+b,alpha); the predictive distribution
# MLE for W, b, and beta
W_ml = X.T @ t / (X.T @ X)  # Normal eq.
b_ml = np.mean(t) - W_ml * np.mean(X)

y = X * W_ml + b_ml

alpha_ml = np.mean((t - y)**2)

plt.plot(X, y, color='red', alpha=0.75, label='Regression line')

# Sample from predictive dist.
ys = np.random.normal(y, alpha_ml)

plt.scatter(X, ys, alpha=0.15, label='Posterior samples')
plt.legend(loc='best')
plt.show()
