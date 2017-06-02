# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:09:30 2017

@author: Ratnadeepb
"""

# Data - (systolic blood pressure, age in years, weight in pounds)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel("mlr02.xls")
X = df.as_matrix()

plt.subplot(211)
plt.scatter(X[:,1], X[:,0])
plt.subplot(212)
plt.scatter(X[:,2], X[:,0])
plt.show()

df['ones'] = 1  # Bias
Y = df['X1']
X = df[['X2', 'X3', 'ones']].values
X2only = df[['X2', 'ones']].values
X3only = df[['X3', 'ones']].values

def get_weights(X, Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def get_Yhat(X, Y):
    return X.dot(get_weights(X, Y))

def RSquared(X, Y):
    d1 = Y - get_Yhat(X, Y)
    d2 = Y - Y.mean()
    return (1 - d1.dot(d1) / d2.dot(d2))

print("X, Y performane: ", RSquared(X, Y))
print("X2, Y performane: ", RSquared(X2only, Y))
print("X3, Y performane: ", RSquared(X3only, Y))

# Adding new dimensions, even if they are not useful will increase
# r-squared
noise = []
for i in range(X.shape[0]):
    r = np.random.randn()
    if r < 0: r = -r
    if r > 0 and r < 0.5: r += 1
    noise.append(r)
noise = np.array(noise)
df["noise"] = noise

Xn = df[['X2', 'X3', "noise", "ones"]].values
print("Xn, Y performane: ", RSquared(Xn, Y))