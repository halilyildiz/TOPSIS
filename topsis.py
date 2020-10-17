# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:25:55 2020

@author: halil
"""

#%% import library

import numpy as np
import pandas as pd

#%% Create a Decision Matrix
a = np.array(pd.read_csv('decision-matrix.csv'))

decission_matrix = a[1:-1,:].astype(float)
weigths = a[0,:].astype(float)
states = a[-1,:].astype(str)

#%% Step 1: Create Normalized Matrix
normalize_matrix = decission_matrix / np.sqrt(np.sum(np.float_power(decission_matrix, 2),axis = 0))

#%% Step 2: Create weighted Normalized Matrix

weigt_normalize_matrix =  normalize_matrix * weigths

#%% Step 3: Determination of Ideal and Negative Solutions
"""
A_positive = np.array([[np.amax(weigt_normalize_matrix[:,i]) if states[i] == "max" else np.amin(weigt_normalize_matrix[:,i]) for i in range(len(states))]])
A_negative = np.array([[np.amin(weigt_normalize_matrix[:,i]) if states[i] == "max" else np.amax(weigt_normalize_matrix[:,i]) for i in range(len(states))]])
"""
A_positive = np.zeros(decission_matrix.shape[1])
A_negative = np.zeros(decission_matrix.shape[1])

for i in range(len(states)):
    if states[i] == "max":
        A_positive[i] = np.amax(weigt_normalize_matrix[:,i])
        A_negative[i] = np.amin(weigt_normalize_matrix[:,i])
    elif states[i] == "min":
        A_positive[i] = np.amin(weigt_normalize_matrix[:,i])
        A_negative[i] = np.amax(weigt_normalize_matrix[:,i])

#%% Step 4: Finding Discrimination Measures
        
S_positive = np.zeros(decission_matrix.shape[0])
S_negative = np.zeros(decission_matrix.shape[0])       

sum_power_positive = np.zeros(decission_matrix.shape[0])
sum_power_negative = np.zeros(decission_matrix.shape[0])

for i in range(weigt_normalize_matrix.shape[1]):
    for j in range(weigt_normalize_matrix.shape[0]):
        sum_power_positive[j] += np.float_power(weigt_normalize_matrix[j,i] - A_positive[i],2)
        sum_power_negative[j] += np.float_power(weigt_normalize_matrix[j,i] - A_negative[i],2)
    
S_positive = np.sqrt(sum_power_positive)
S_negative = np.sqrt(sum_power_negative)

#%% Step 5: Computing Relative Proximity to Ideal Solution

G = np.zeros(decission_matrix.shape[0])
for i in range(weigt_normalize_matrix.shape[0]):
    G[i] = S_negative[i] /(S_positive[i] + S_negative[i])
    
#%% Result
    alternative = np.zeros(decission_matrix.shape[0]).astype(str)
    
    for i in range(decission_matrix.shape[0]):
        alternative[i] = ("A{}".format(i+1))
        
    result = pd.DataFrame(G, index = alternative, columns = ["G"] )
    
    result.sort_values(by=['G'],ascending = False, inplace = True)
    result.to_csv('result.csv',encoding = 'utf-8')
    