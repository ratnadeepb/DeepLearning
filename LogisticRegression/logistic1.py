#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:53:50 2017

@author: ratnadeepb
"""

import numpy as np

N = 100
D = 2

X = np.random.randn(100, 2)

ones = np.array([[1] * N]).T

Xb = np.concatenate((ones, X), axis=1)
w = np.random.randn(D + 1)

z = Xb.dot(w)


def sigmoid(x, w):
    return 1 / (1 - np.exp(- (x.dot(w))))

print(sigmoid(Xb, w))