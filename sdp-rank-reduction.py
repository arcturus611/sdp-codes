#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:45:03 2019

@author: arcturus
"""
#%% NOTES: 
# TBD 1: test rank reduction with a random sdp (start w/ a matrix 
# and generate constraints based on that.)
# NOTE - This is awful- the solver returns a matrix that has a TINY negative eigenvalue!?
# NOTE 2 - The rank reducer requires Cholesky, whch requires X to be PD, not PSD. Do I just add an eps perturbation?
#%% Rank reduction alg
import numpy as np
import scipy as sp
import cvxpy as cp

#%% Generate a random symmetric matrix.
def generate_rndsymm(n):
    C = np.random.rand(n, n)
    C = (C + np.transpose(C))*0.5
    return C

#%% Generate a random PSD matrix; the PSDness follows from Theorem 6.1.10 of HJ
# (diagonal dominance + diagonals non-negative + symmetry).
def generate_psd(s):
    A = generate_rndsymm(s)
    Apsd = A + s*np.eye(s)
    return Apsd
    
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
    [r, n] = A.shape
    m = r//n
    
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
    print("Length of delta is ",np.size(delta))
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
def sdp(_test_): 
    n = _test_.num_points
    m = _test_.num_constraints
    cons = _test_.constraints
    obj = _test_.objective
    
    X = cp.Variable((n, n), symmetric = True)
    constraints = [X>>0]
    constraints+= [cp.trace(cons.A[i]@X) == cons.b[i] for i in range(m)]
    prob = cp.Problem(cp.Minimize(cp.trace(obj@X)), constraints)
    prob.solve()
    return prob, X

#%% 
class Constraints:
    def __init__(self, A, b):
        self.A = A
        self.b = b

#%% 
class TestSDP:
    def __init__(self, n, m):
        self.num_points = n
        self.num_constraints = m
        self.objective = self.generate_obj()
        self.constraints, self.one_feasible_point = self.generate_constraints()
    
    def generate_obj(self):
        n = self.num_points
        C = generate_psd(n) # so that the obj doesn't become -inf when minimizing
        return C
    
    def generate_constraints(self):
        n = self.num_points
        m = self.num_constraints
        
        A = []
        for i in range(m):
            A.append(generate_rndsymm(n))
        #A_cat = np.vstack(A[i] for i in range(m))  
        
        X_feas = generate_psd(n) 
        
        b = []
        for i in range(m):
            b.append(np.trace(A[i].dot(X_feas)))
        #b_cat = np.vstack(b[i] for i in range(m))
        
        constraints = Constraints(A, b)
        return constraints, X_feas
    
        def check_feasibility(self, X):
            #TBD
            return
#%% 
def check_pd(X):
    lambda_min = np.min(np.linalg.eigvals(X))
    eps = 1e-06
    
    if(lambda_min > 0):
        return X
    else:
        return X + np.eye(X.shape[0])*(np.abs(lambda_min) + eps)

#%%
if __name__ == '__main__':
    n = 5
    m = 3
    _testsdp_ = TestSDP(n, m)
    _testsdpsol_, _testsdpX_ = sdp(_testsdp_)
    A_cat = np.vstack(_testsdp_.constraints.A[i] for i in range(m))
    pdfied_X = check_pd(_testsdpX_.value)
    #_testsdp_.check_feasibility(pdfied_X)
    rank_reduction(pdfied_X, A_cat)
    
                