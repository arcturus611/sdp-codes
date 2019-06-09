#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:35:22 2019

@author: arcturus
"""
#%% import libraries
import cvxpy as cp
import numpy as np
import scipy as sp
from os import system

#%% clear screen 
_ = system('clear')

#%% generate data
m = 5 #number of constraints
n = 3 #dimensions of matrix var
#np.random.seed(1)

C = np.random.randn(n, n) #constraint matrix
A = [] #empty list
b = []

for i in range(m): #generate all the m constraints
    A.append(np.random.randn(n, n))
    b.append(np.random.randn())

#%% define the problem
X = cp.Variable((n, n), symmetric = True) #init the program vars
constraints = [X >> 0] #psd 
# equality constraints (note the conditional (not assignment) operator)
constraints+= [cp.trace(A[i]@X) == b[i] for i in range(m)] 

#%% call the solver and output the result
prob = cp.Problem(cp.Minimize(cp.trace(C@X)), constraints)
prob.solve()
print('The optimal value is ', prob.value)
print('The optimizer is ', X.value)
print('The rank of the optimizer is ', np.linalg.matrix_rank(X.value))
