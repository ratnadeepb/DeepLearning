#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 21:37:45 2017

@author: ratnadeepb
"""

import numpy as np
import pandas as pd

def get_data():
    # Get the data
    df = pd.read_csv("ecommerce_data.csv")
    data = df.values

    # Divide into X and Y
    X = data[:-1]
    Y = data[-1]

    # Normalise the numerical variables
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    # Handle the categorical variables
    N, D = X.shape
    X2 = np.zeros((N, D+3))  # There are 3 different categories
    X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]

    # For the last 3 columns
    # One-Hot Encoding
    for n in range(N):
        t = int(X[n, (D - 1)])
        X2[n, t+D-1] = 1

    return X2, Y

def get_binary_data():
    # Pick up binary data only
    X, Y = get_data()
    X2 = X[np.where(Y <= 1)]
    Y2 = Y[np.where(Y <= 1)]

    return X2, Y2