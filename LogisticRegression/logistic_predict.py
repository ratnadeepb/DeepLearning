#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Created on 20 Jun 20:44 2017"
"@author: ratnadeepb"

import numpy as np
from process import get_binary_data


X, Y = get_binary_data()

D = X.shape[1]

W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b)

predictions = np.round(P_Y_given_X)

def classification_rate(Y, P):
    return np.mean(Y == P)

print("Classification rate: ", classification_rate(Y, P_Y_given_X))