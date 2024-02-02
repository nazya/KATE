#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

### Data Generation

### Initialize the number of features and data points
n_feature = 20   # dimension of data
n_data = 1000    # number of data-points
n_iteration = int(1e5)  
n_trial = 1
n_epoch = 10
S = 10      # size of mini batch

### Initialize the data matrix and optimal weight
np.random.seed(0)
X = np.random.normal(0, 1, (n_data, n_feature))/np.sqrt(n_feature)
w_optimal = np.random.normal(0, 10, n_feature)

### Generate the scaling diagonal matrix V such that each of the elements is e^r where r is generated from Uniform(-10, 10)
V = np.diag(np.exp(np.random.uniform(-10, 10, n_feature)))
V_inv_sq = np.linalg.matrix_power(np.linalg.inv(V), 2) 
### Scale the data matrix using V
X_scaled = X @ V

### Fix the array of labels y
y = np.sign(X_scaled @ w_optimal)
y[y == 0] = 1    # Note that np.sign(0) = 0. So we set the label y = 1 in case any of the y is 0.


### Define Functions

### Define F
def F(X, y, w,lmd1=0):
  r = -y*X.dot(w)
  expr = np.exp(r);
  return np.mean(np.log(1+expr))+ 0.5 * lmd1 * np.linalg.norm(w)**2

### Define Grad F
def dF(X, y, w,lmd1=0):
  r = -y*X.dot(w)
  expr = np.exp(r);
  grad= X.T.dot(-expr/(1+expr)*y)   
  return grad + lmd1 * w



### Choose hyperparameters
beta = 1e-2
eta = 1/((dF(X,y,np.zeros(n_feature))/n_data)**2)
etaScaled = 1/((dF(X_scaled,y,np.zeros(n_feature))/n_data)**2)
# eta, etaScaled = 0, 0

### Initiaalize 
w = np.random.randn(n_feature)*0.0    # intialize w = 0 
wScaled = np.random.randn(n_feature)*0.0    # intialize w = 0 

gScaled_sum = 0
gScaled_normsum = 0

g_sum = 0
g_normsum = 0

KATEhist = []
KATEhistScaled = []

for epo in range(n_epoch):
    for it in range(n_data):
        i = np.random.randint(n_data)
        j = np.minimum(n_data, i+S)
        y_batch = y[i:j]  # batch label
        
        # for scaled data
        X_batch = X_scaled[i:j,:]  # batch Scaled data
        g = dF(X_batch,y_batch,wScaled)/S # gradient of scaled data
        gScaled_sum += g*g
        gScaled_normsum += (g*g)/gScaled_sum
        wScaled -= (beta * np.sqrt(etaScaled * gScaled_sum + gScaled_normsum)) * (g/gScaled_sum)
        KATEhistScaled.append((F(X_scaled, y, wScaled), (dF(X_scaled, y, wScaled)/n_data).dot(V_inv_sq @ (dF(X_scaled, y, wScaled)/n_data)), np.sum([y*X_scaled.dot(wScaled) >= 0])/n_data))
        
        X_batch = X[i:j,:]
        g = dF(X_batch, y_batch, w)/S
        g_sum += g*g
        g_normsum += (g*g)/g_sum
        w -= (beta * np.sqrt(eta * g_sum + g_normsum)) * (g/g_sum)
        KATEhist.append((F(X, y, w), np.linalg.norm(dF(X, y, w)/n_data)**2, np.sum([y*X.dot(w) >= 0])/n_data))
        
    

### Make a single plot for f and grad f and Accuracy
marker = np.arange(0, n_epoch * n_data, (n_epoch * n_data)/100, dtype='int')

plt.figure()
plt.plot(marker, [KATEhist[i][0] for i in marker], color = 'b', label = r'Dataset: $(x_i,y_i)$')
plt.plot(marker, [KATEhistScaled[i][0] for i in marker], color = 'orange', linestyle = 'dashed', label = r'Dataset: $(Vx_i,y_i)$')
plt.grid(True)
plt.ylabel(r'$f(w_t)$', fontsize = 15)
plt.xlabel('iterations', fontsize = 15)
plt.legend(fontsize = 15)
plt.savefig(r'scale_functionval.pdf')

plt.figure()
plt.plot(marker, [KATEhist[i][1] for i in marker], color = 'b', label = r'$\Vert \nabla f(w_t) \Vert^2$; Dataset: $(x_i,y_i)$')
plt.plot(marker, [KATEhistScaled[i][1] for i in marker], color = 'orange', linestyle = 'dashed', label = r'$\Vert \nabla f(w_t) \Vert^2_{V^{-2}}$; Dataset: $(Vx_i,y_i)$')
plt.yscale('log')
plt.grid(True)
plt.ylabel(r'Grad Norm', fontsize = 15)
plt.xlabel('iterations', fontsize = 15)
plt.legend(fontsize = 15)
plt.savefig(r'scale_grad.pdf')

plt.figure()
plt.plot(marker, [KATEhist[i][2] for i in marker], color = 'b', label = r'Dataset: $(x_i,y_i)$')
plt.plot(marker, [KATEhistScaled[i][2] for i in marker], color = 'orange', linestyle = 'dashed', label = r'Dataset: $(Vx_i,y_i)$')
plt.grid(True)
plt.ylabel(r'Accuracy', fontsize = 15)
plt.xlabel('iterations', fontsize = 15)
plt.legend(fontsize = 15)
plt.savefig(r'scale_accuracy.pdf')


