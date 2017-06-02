# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:43:52 2017

@author: Ratnadeepb
"""

import numpy as np
import matplotlib.pyplot as plt

# Functions
def get_weights(X, Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def get_Yhat(X, Y):
    return X.dot(get_weights(X, Y))

def RSquared(X, Y):
    d1 = Y - get_Yhat(X, Y)
    d2 = Y - Y.mean()
    return (1 - d1.dot(d1) / d2.dot(d2))

# Generating random data
N = 50

X = np.linspace(0, 10, N)
Y = 0.5 * X + np.random.randn()

# Generating outliers
Y[-1] += 30
Y[-2] += 50

# Adding bias
X = np.vstack([np.ones(N), X]).T

# Ridge Regression
lam = 1000
w_rr = np.linalg.solve(lam * np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_rr = X.dot(w_rr)

plt.scatter(X[:,1], Y, label="data")
plt.plot(X[:,1], get_Yhat(X, Y), label="mle")
plt.plot(X[:,1], Yhat_rr, label="ridge")
plt.legend()
plt.show()
# print("Maximum likelihood r-squared: ", RSquared(X, Y))