#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:45:03 2019

@author: arcturus
"""

#%% Rank reduction alg
import numpy as np
import scipy as sp
import cvxpy as cp
import mybasicsmodule

#%% Construct the linear operator Av
def create_Av(A, V):
    #inputs: 
    # A = mn by n matrix formed by vertically stacking all the A_i s
    # V = n by r matrix

    #outputs: 
    # Av = m by r^2 matrix obtained by ...
    # first computing all the m products V^T A_i V (an r by r matrix),
    # then vectorizing each of them, 
    # and vertically stacking the row vectors. 
    [r, n] = A.shape()
    m = r/n
    
    AV = A.dot(V) #mn by r matrix 
    AVs = np.vsplit(AV, m) #list of m arrays of size n by r
    
    VtAV_listmat = [V.transpose().dot(AVs[_]) for _ in range(m)] #
    
    VtAV_listvec = [VtAV_listmat[_].flatten() for _ in range(m)]
    Av = np.asarray(VtAV_listvec)
    return Av
#%%        
def create_nullspace_Av(Av):
    #inputs: 
    # Av = m by r^2 matrix, the i'th row being vec(V' A_i V)
    
    #outputs:
    # Delta = r by r matrix obtained as follows: find a vector delta 
    # in the nullspace of Av; write this as a matrix (row-wise).
    
    delta = sp.linalg.null_space(Av)
    r = np.sqrt(np.size(delta))
    Delta = np.reshape(delta, (r, r))
    return Delta
#%%
def rank_reduction(X, A):
    #inputs:
    # X = n by n matrix, solution of given SDP
    # A = mn by n matrix, stack of m constraint matrices
    
    #outputs:
    # X = n by n matrix; solution to the SDP that has a rank <= that of the input
    
    # cholesky factorize X to get V
    V = np.linalg.cholesky(X)
    r = V.shape[0]
    I_r = np.eye(r)
    Av = create_Av(A, V) 
    
    Delta = create_nullspace_Av(Av)
    while ( Delta.size != 0):
        # Find lambda_1 which is the max eigval of Delta
        lambda_1 = np.max(np.abs(np.linalg.eigvals(Delta)))
        # Choose alpha according to formula
        alpha = -1/lambda_1
        # Construct X = V(I + alpha * Delta)V'
        temp = (I_r + alpha*Delta).dot(V.transpose())
        X = V.dot(temp)
        # find Delta in nullspace(A_v)
        Delta = create_nullspace_Av(Av)

    return X

#%% 
if __name__ == '__main__':
        C = mybasicsmodule.generate_psd(5)
        X = cp.Variable((5, 5), symmetric = True)
        constraints = [X >> 0]
        constraints+= [cp.trace(X)==1]
        prob = cp.Problem(cp.Maximize(cp.trace(C@X)), constraints)
        prob.solve()
        print("The problem\'s optimal value is ", prob.value)
        print("The optimizer is ", X.value)