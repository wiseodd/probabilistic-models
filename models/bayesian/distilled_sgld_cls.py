"""
Bayesian Dark Knowledge
-----------------------
Korattikara, et. al., 2015
"""
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]

N = mnist.train.images.shape[0]  # training set size
M = 100  # minibatch size
T = 10000  # num of iteration

Theta = np.vstack([np.zeros([1, y_dim]), np.random.randn(X_dim, y_dim) * 0.001])
W = np.vstack([np.zeros([1, y_dim]), np.random.randn(X_dim, y_dim) * 0.001])


def softmax(x):
    m = np.max(x)
    return np.exp(x - m) / np.sum(np.exp(x - m))


# SGLD Teacher-Student distillation
# ---------------------------------

eta = 0.1  # teacher learning rate
rho = 0.1  # student learning rate
lam = 1  # teacher prior
gam = 1e-3  # student prior

burnin = 1000
thinning = 10

for t in range(1, T+1):
    # Train teacher
    # -------------
    X_mb, y_mb = mnist.train.next_batch(M)
    X_train = np.hstack([X_mb, np.ones([M, 1])])  # add bias dim

    eta_t = eta/t

    p = softmax(X_train @ Theta)
    grad_p = p - y_mb

    grad_loglik = X_train.T @ grad_p  # 784x16 . 16x10
    grad_logprior = lam * Theta
    grad_logpost = grad_logprior + N/M * grad_loglik

    z = np.random.normal(0, np.sqrt(eta_t))
    delta = eta_t/2 * grad_logpost + z
    Theta += delta

    if t > burnin and t % thinning == 0:
        # Train student
        # -------------
        X_s_mb = X_mb + np.random.normal(0, 1e-3)  # perturb
        X_s_train = np.hstack([X_s_mb, np.ones([M, 1])])  # add bias dim
        rho_t = rho/t

        s = softmax(X_s_train @ W)
        p = softmax(X_s_train @ Theta)
        grad_s = s - p

        grad_loglik = X_s_train.T @ grad_s  # 784x16 . 16x10
        grad_logprior = gam * W
        grad_logpost = grad_logprior + 1/M*grad_loglik

        delta = rho_t * grad_logpost
        W -= delta

    # Diagnostics
    # -----------
    if t % 1000 == 0:
        s = softmax(X_train @ W)
        loss = -gam/2 * np.sum(W**2) - np.sum(y_mb * np.log(s + 1e-8))
        print('Iter: {}; S_loss: {:.4f}'.format(t, loss))


# Test
# ----

X_test, y_test = mnist.test.images, mnist.test.labels
X_test = np.hstack([X_test, np.ones([X_test.shape[0], 1])])  # add bias dim

y = softmax(X_test @ W)
acc = np.mean(y.argmax(axis=1) == y_test.argmax(axis=1))

print('Test accuracy: {:.4f}'.format(acc))
