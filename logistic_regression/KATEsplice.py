#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from utils import prepare_data
from scipy.sparse.linalg import svds

X, y, n_data, n_feature = prepare_data("splice")

n_iteration = int(5*1e3)  
n_trial = 5
S = 10      # size of mini batch


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


sigmas = svds(X, return_singular_vectors=False)
m = X.shape[0]
L = sigmas.max()**2 / (4*m)

w_star = np.zeros(n_feature)
for _ in range(1000000):
    w_star -= dF(X, y, w_star)/(L * n_data)


# ### Uncomment the following code to tune the stepsizes

# beta_list = [1e-10, 1e-8, 1e-7, 1e-6, 1e-4, 1e-2, 1]

# FvalKATEhist = {}
# accuracyKATEhist = {}
# eta = 1/((dF(X,y,np.zeros(n_feature))/n_data)**2)
# #eta = 1

# for beta in beta_list:
#     FvalKATEhist[beta] = []
#     accuracyKATEhist[beta] = []
#     for trial in range(n_trial):
#         trialFval = []
#         trialaccuracy = []
#         w = np.random.randn(n_feature)*0.0

#         # Initialize hyperparamter beta
#         g_sum = 1e-6       # this stores \sum_i g_i * g_i (coordinate-wise multiplication) 
#         g_normsum = 0          # this stores \sum_i norm(g_i)**2 
#         for it in range(n_iteration):
#             i = np.random.randint(n_data)
#             j = np.minimum(n_data, i+S)
#             y_batch = y[i:j]  # batch label
#             # for scaled data
#             X_batch = X[i:j,:]  # batch Scaled data
#             g = dF(X_batch,y_batch,w)/S # gradient of scaled data
#             g_sum += g*g
#             g_normsum += (g * g)/g_sum
#             w -= (beta * np.sqrt(eta * g_sum + g_normsum))*(g/g_sum)    # update step
#             trialFval.append(F(X, y, w))
#             # trialaccuracy.append(np.sum([y_test*X_test.dot(w) >= 0])/n_data_test)
        
#         FvalKATEhist[beta].append(trialFval)
#         accuracyKATEhist[beta].append(trialaccuracy)

# for beta in beta_list:
#     print(beta, np.mean(FvalKATEhist[beta], axis = 0))        
        
# FvalAdaGradhist = {}
# accuracyAdaGradhist = {}
# for beta in beta_list:
#     FvalAdaGradhist[beta] = []
#     accuracyAdaGradhist[beta] = []
#     for trial in range(n_trial):
#         trialFval = []
#         trialaccuracy = []
#         w = np.random.randn(n_feature)*0.0

#         # Initialize hyperparamter beta
#         g_sum = 1e-6        # this stores \sum_i g_i * g_i (coordinate-wise multiplication) 
        
#         for it in range(n_iteration):
#             i = np.random.randint(n_data)
#             j = np.minimum(n_data, i+S)
#             y_batch = y[i:j]  # batch label
            
#             # for scaled data
#             X_batch = X[i:j,:]  # batch Scaled data
#             g = dF(X_batch,y_batch,w)/S # gradient of scaled data
#             g_sum += g*g
#             w -= beta*(g/np.sqrt(g_sum))    # update step
            
#             trialFval.append(F(X, y, w))
#             # trialaccuracy.append(np.sum([y_test*X_test.dot(w) >= 0])/n_data_test)
#         FvalAdaGradhist[beta].append(trialFval)
#         accuracyAdaGradhist[beta].append(trialaccuracy)

# for beta in beta_list:
#     print(beta, np.mean(FvalAdaGradhist[beta], axis = 0))
        

beta = 1e-2
eta = 1/((dF(X,y,np.zeros(n_feature))/n_data)**2)
FvalKATEhist = []
accuracyKATEhist = []

for trial in range(n_trial):
    trialFval = []
    trialaccuracy = []
    w = np.random.randn(n_feature)*0.0

    # Initialize hyperparamter beta
    g_sum = 1e-6       # this stores \sum_i g_i * g_i (coordinate-wise multiplication) 
    g_normsum = 0          # this stores \sum_i norm(g_i)**2 
    for it in range(n_iteration):
        trialFval.append(F(X, y, w))
        trialaccuracy.append(np.sum([y*X.dot(w) > 0])/n_data)
        
        i = np.random.randint(n_data)
        j = np.minimum(n_data, i+S)
        y_batch = y[i:j]  # batch label
        # for scaled data
        X_batch = X[i:j,:]  # batch Scaled data
        g = dF(X_batch,y_batch,w)/S # gradient of scaled data
        g_sum += g*g
        g_normsum += (g * g)/g_sum
        w -= (beta * np.sqrt(eta * g_sum + g_normsum))*(g/g_sum)    # update step
        
    
    FvalKATEhist.append(trialFval)
    accuracyKATEhist.append(trialaccuracy)
    
    
beta = 1e-2
FvalAdaGradhist = []
accuracyAdaGradhist = []
for trial in range(n_trial):
    trialFval = []
    trialaccuracy = []
    w = np.random.randn(n_feature)*0.0

    # Initialize hyperparamter beta
    g_sum = 1e-6        # this stores \sum_i g_i * g_i (coordinate-wise multiplication) 
    
    for it in range(n_iteration):
        trialFval.append(F(X, y, w))
        trialaccuracy.append(np.sum([y*X.dot(w) > 0])/n_data)
        i = np.random.randint(n_data)
        j = np.minimum(n_data, i+S)
        y_batch = y[i:j]  # batch label
        
        # for scaled data
        X_batch = X[i:j,:]  # batch Scaled data
        g = dF(X_batch,y_batch,w)/S # gradient of scaled data
        g_sum += g*g
        w -= beta*(g/np.sqrt(g_sum))    # update step
        

    FvalAdaGradhist.append(trialFval)
    accuracyAdaGradhist.append(trialaccuracy)

beta = 1e-2   
FvalAdaGradNormhist = []
accuracyAdaGradNormhist = []
for trial in range(n_trial):
    trialFval = []
    trialaccuracy = []
    w = np.random.randn(n_feature)*0.0

    # Initialize hyperparamter beta
    g_sum = 1e-6        # this stores \sum_i g_i * g_i (coordinate-wise multiplication) 
    
    for it in range(n_iteration):
        trialFval.append(F(X, y, w))
        trialaccuracy.append(np.sum([y*X.dot(w) > 0])/n_data)
        i = np.random.randint(n_data)
        j = np.minimum(n_data, i+S)
        y_batch = y[i:j]  # batch label
        
        # for scaled data
        X_batch = X[i:j,:]  # batch Scaled data
        g = dF(X_batch,y_batch,w)/S # gradient of scaled data
        g_sum += np.linalg.norm(g)**2
        w -= beta*(g/np.sqrt(g_sum))    # update step
        

    FvalAdaGradNormhist.append(trialFval)
    accuracyAdaGradNormhist.append(trialaccuracy)
    


beta = 1e-4   
FvalSGDdecayhist = []
accuracySGDdecayhist = []
for trial in range(n_trial):
    trialFval = []
    trialaccuracy = []
    w = np.random.randn(n_feature)*0.0
    
    for it in range(n_iteration):
        trialFval.append(F(X, y, w))
        trialaccuracy.append(np.sum([y*X.dot(w) > 0])/n_data)
        i = np.random.randint(n_data)
        j = np.minimum(n_data, i+S)
        y_batch = y[i:j]  # batch label
        
        # for scaled data
        X_batch = X[i:j,:]  # batch Scaled data
        g = dF(X_batch,y_batch,w)/S # gradient of scaled data
        w -= beta*(g/np.sqrt(it+1))    # update step
        

    FvalSGDdecayhist.append(trialFval)
    accuracySGDdecayhist.append(trialaccuracy)
    
    

beta = 1e-4   
FvalSGDconstanthist = []
accuracySGDconstanthist = []
for trial in range(n_trial):
    trialFval = []
    trialaccuracy = []
    w = np.random.randn(n_feature)*0.0
    
    for it in range(n_iteration):
        trialFval.append(F(X, y, w))
        trialaccuracy.append(np.sum([y*X.dot(w) > 0])/n_data)
        i = np.random.randint(n_data)
        j = np.minimum(n_data, i+S)
        y_batch = y[i:j]  # batch label
        
        # for scaled data
        X_batch = X[i:j,:]  # batch Scaled data
        g = dF(X_batch,y_batch,w)/S # gradient of scaled data
        w -= beta*g    # update step
        
    FvalSGDconstanthist.append(trialFval)
    accuracySGDconstanthist.append(trialaccuracy)
    


plt.figure()
marker = np.arange(0, n_iteration, (n_iteration)/10, dtype='int')
plt.plot(marker, np.mean(FvalKATEhist, axis =0)[marker], color = 'b', marker = 'o', label = 'KATE')
plt.plot(marker, np.mean(FvalAdaGradhist, axis = 0)[marker], color = 'r', marker = '>', label = 'AdaGrad')
plt.plot(marker, np.mean(FvalAdaGradNormhist, axis = 0)[marker], color = 'g', linestyle = 'dashed' , marker = 'p', label = 'AdaGradNorm')
plt.plot(marker, np.mean(FvalSGDconstanthist, axis = 0)[marker], color = 'm', marker = 'v', label = 'SGD-constant')
plt.plot(marker, np.mean(FvalSGDdecayhist, axis = 0)[marker], color = 'c', marker = 's', label = 'SGD-decay')
plt.yscale('log')
plt.grid(True)
plt.ylabel(r'$f(w_t)$', fontsize = 15)
plt.xlabel('iterations', fontsize = 15)
plt.legend()
plt.savefig(r'splice_fval.pdf')

plt.figure()
marker = np.arange(0, n_iteration, (n_iteration)/10, dtype='int')
plt.plot(marker, np.mean(accuracyKATEhist, axis =0)[marker], color = 'b', marker = 'o', label = 'KATE')
plt.plot(marker, np.mean(accuracyAdaGradhist, axis = 0)[marker], color = 'r', marker = '>', label = 'AdaGrad')
plt.plot(marker, np.mean(accuracyAdaGradNormhist, axis = 0)[marker], color = 'g', linestyle = 'dashed' , marker = 'p', label = 'AdaGradNorm')
plt.plot(marker, np.mean(accuracySGDconstanthist, axis = 0)[marker], color = 'm', marker = 'v', label = 'SGD-constant')
plt.plot(marker, np.mean(accuracySGDdecayhist, axis = 0)[marker], color = 'c', marker = 's', label = 'SGD-decay')
plt.grid(True)
plt.ylabel(r'Accuracy', fontsize = 15)
plt.xlabel('iterations', fontsize = 15)
plt.legend()
plt.savefig(r'splice_accuracy.pdf')
    
    
