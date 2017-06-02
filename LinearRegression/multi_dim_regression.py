# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:38:50 2017

@author: Ratnadeepb
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load data
X = []
Y = []

with open("data_2d.csv",'r') as data:
    for line in data:
        x1, x2, y = line.split(',')
        X.append([float(x1), float(x2), 1])
        Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()


# Calculate weights
w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

Yhat = X.dot(w)

# R-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("The R-squared score is: ", r2)
