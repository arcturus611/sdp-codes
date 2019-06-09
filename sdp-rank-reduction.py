#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:45:03 2019

@author: arcturus
"""

#$$ Rank reduction alg
import numpy as np
import scipy as sp
import cvxpy as cp

def create_Av(A, V):
    
def create_nullspace_Av(Av):
    

def rank_reduction(X, A):
    # cholesky factorize X to get V
    V = np.linalg.cholesky(X)
    r = V.shape[0]
    I_r = np.eye(r)
    Av = create_Av(A, V) 
    
    Delta = sp.linalg.null_space(Av) # this is actually a bit more complicated- Av isn't a matrix, just a lin op
    while ( Delta.size != 0):
    # Find lambda_1 which is the max eigval of Delta
        lambda_1 = np.linalg.eigvals(Delta)[0]
    # Choose alpha according to formula
        alpha = -1/lambda_1
    # Construct X = V(I + alpha * Delta)V'
        temp = (I_n + alpha*Delta)*V.transpose()
        X = V*temp
    # find Delta in nullspace(A_v)
        Delta = sp.linalg.null_space(Av)
