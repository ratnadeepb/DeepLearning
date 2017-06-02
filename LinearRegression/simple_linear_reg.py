# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:01:04 2017

@author: Ratnadeepb
"""

# Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt

def slr(data):
    assert isinstance(data, np.ndarray), "We need Numpy arrays only"
    assert data.shape[1] == 2, "This is 1-D Linear Regression"
    
    x = data[:, 0]
    y = data[:, 1]
    
    xy_bar = x.dot(y) / len(x)
    x_bar = x.sum() / len(x)
    y_bar = y.sum() / len(y)
    x2_bar = x.dot(x) / len(x)
    
    slope = (xy_bar - (x_bar) * (y_bar)) / (x2_bar - (x_bar) ** 2)
    intercept = (x2_bar * y_bar - x_bar * xy_bar) / (x2_bar - (x_bar) ** 2)
    return (intercept, slope)

def plot_data(data):
    assert isinstance(data, np.ndarray), "We need Numpy arrays only"
    assert data.shape[1] == 2, "This is 1-D Linear Regression"
    
    intercept, slope = slr(data)
    x = data[:, 0]
    y = data[:, 1]
    
    plt.scatter(x, y)
    plt.plot(x, (x * slope) + intercept)
    plt.legend(['Best Fit', 'X'])
    plt.show()
    return (intercept, slope)
    

if __name__ == "__main__":
    data = []
    with open('data_1d.csv', 'r') as d:
        for line in d.readlines():
            x, y = line.split(',')
            data.append([float(x), float(y)])
    data = np.array(data)
    intercept, slope = plot_data(data)
    print("The intercept is: {} and the slope is: {}".format(intercept, slope))