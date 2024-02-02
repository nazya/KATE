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


"""
Define Functions

"""

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

"""
Set F_star = F(w^*)

"""
F_star = F(X_scaled, y, w_optimal)  # computes the optimal value of F at w_optimal


"""
List of initialization

"""
b0 = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]


"""
Implement KATE
"""
FvalKATEhist = {}
eta = 1

for b0index in b0:
    FvalKATEhist[b0index] = []
    
    for trial in range(n_trial):
        trialFval = []
        
        w = np.random.randn(n_feature)*0.0

        # Initialize hyperparamter beta
        beta = F(X_scaled, y, w) - F_star
        g_sum = b0index        # this stores \sum_i g_i * g_i (coordinate-wise multiplication) 
        g_normsum = 0          # this stores \sum_i norm(g_i)**2 
        for it in range(n_iteration):
            i = np.random.randint(n_data)
            j = np.minimum(n_data, i+S)
            y_batch = y[i:j]  # batch label
            # for scaled data
            X_batch = X_scaled[i:j,:]  # batch Scaled data
            g = dF(X_batch,y_batch,w)/S # gradient of scaled data
            g_sum += g*g
            g_normsum += (g * g)/g_sum
            w -= (beta * np.sqrt(eta * g_sum + g_normsum))*(g/g_sum)    # update step
            trialFval.append(F(X_scaled, y, w))
        
        FvalKATEhist[b0index].append(trialFval)
   
        
"""
Implement AdaGrad
"""

FvalAdaGradhist = {}
for b0index in b0:
    FvalAdaGradhist[b0index] = []
    
    for trial in range(n_trial):
        trialFval = []
        w = np.random.randn(n_feature)*0.0

        # Initialize hyperparamter beta
        beta = F(X_scaled, y, w) - F_star
        g_sum = b0index        # this stores \sum_i g_i * g_i (coordinate-wise multiplication) 
        
        for it in range(n_iteration):
            i = np.random.randint(n_data)
            j = np.minimum(n_data, i+S)
            y_batch = y[i:j]  # batch label
            
            # for scaled data
            X_batch = X_scaled[i:j,:]  # batch Scaled data
            g = dF(X_batch,y_batch,w)/S # gradient of scaled data
            g_sum += g*g
            w -= beta*(g/np.sqrt(g_sum))    # update step
            
            trialFval.append(F(X_scaled, y, w))
        FvalAdaGradhist[b0index].append(trialFval)
        
"""
Implement AdaGradNorm
"""

FvalAdaGradNormhist = {}
for b0index in b0:
    FvalAdaGradNormhist[b0index] = []
    
    for trial in range(n_trial):
        trialFval = []
        w = np.random.randn(n_feature)*0.0
        
        beta = F(X_scaled, y, w) - F_star
        g_sum = b0index
        
        for it in range(n_iteration):
            i = np.random.randint(n_data)
            j = np.minimum(n_data, i+S)
            y_batch = y[i:j]
            
            X_batch = X_scaled[i:j,:]
            g = dF(X_batch, y_batch, w)/S
            g_sum += np.linalg.norm(g)**2
            w -= (beta/np.sqrt(g_sum))*g 
            trialFval.append(F(X_scaled, y, w))
        FvalAdaGradNormhist[b0index].append(trialFval)
        

"""
Implement SGD-decay
"""

FvalSGDdecayhist = {}
for b0index in b0:
    FvalSGDdecayhist[b0index] = []
    
    for trial in range(n_trial):
        trialFval = []
        
        w = np.random.randn(n_feature)*0.0

        # Initialize hyperparamter beta
        beta = F(X_scaled, y, w) - F_star
        
        for it in range(n_iteration):
            i = np.random.randint(n_data)
            j = np.minimum(n_data, i+S)
            y_batch = y[i:j]  # batch label
            
            # for scaled data
            X_batch = X_scaled[i:j,:]  # batch Scaled data
            g = dF(X_batch,y_batch,w)/S # gradient of scaled data
            w -= (beta/(b0index * np.sqrt(it + 1)))*g    # update step
            
            trialFval.append(F(X_scaled, y, w))
        FvalSGDdecayhist[b0index].append(trialFval)
        
     
"""
Implement SGD-constant
"""

FvalSGDconstanthist = {}
for b0index in b0:
    FvalSGDconstanthist[b0index] = []
    for trial in range(n_trial):
        trialFval = []
        w = np.random.randn(n_feature)*0.0
        
        beta = F(X_scaled, y, w) - F_star
        step = beta/b0index
        for it in range(n_iteration):
            i = np.random.randint(n_data)
            j = np.minimum(n_data, i+S)
            y_batch = y[i:j]
            
            X_batch = X_scaled[i:j,:]
            g = dF(X_batch, y_batch, w)/S
            w -= step*g
            trialFval.append(F(X_scaled, y, w))
        FvalSGDconstanthist[b0index].append(trialFval)


"""
Plots
"""

for compute_iter in [int(1e4), int(5*1e4), int(1e5)]:
    plt.figure()
    plt.plot(b0[1:], [np.mean([FvalKATEhist[b0index][trial][compute_iter - 1] for trial in range(n_trial)]) for b0index in b0[1:]], color = 'b', marker = 'o', label = 'KATE')
    plt.plot(b0[1:], [np.mean([FvalAdaGradhist[b0index][trial][compute_iter - 1] for trial in range(n_trial)]) for b0index in b0[1:]], color = 'r', marker = '>', label = 'AdaGrad')
    plt.plot(b0[1:], [np.mean([FvalAdaGradNormhist[b0index][trial][compute_iter - 1] for trial in range(n_trial)]) for b0index in b0[1:]], color = 'g', linestyle = 'dashed' , marker = 'p', label = 'AdaGradNorm')
    plt.plot(b0[1:], [np.mean([FvalSGDconstanthist[b0index][trial][compute_iter - 1] for trial in range(n_trial)]) for b0index in b0[1:]], color = 'm', marker = 'v', label = 'SGD-constant')
    plt.plot(b0[1:], [np.mean([FvalSGDdecayhist[b0index][trial][compute_iter - 1] for trial in range(n_trial)]) for b0index in b0[1:]], color = 'c', marker = 's', label = 'SGD-decay')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.ylabel(r'$f(w_t)$', fontsize = 15)
    plt.xlabel(r'$\Delta$', fontsize = 15)
    plt.legend()
    plt.savefig(f'{compute_iter}iteration.pdf')
    
    