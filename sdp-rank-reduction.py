#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:45:03 2019

@author: arcturus
"""
# We implement Algorithm 2.1 described in "Low Rank SDPs: Theory and Applications"
# by Lemon, Man-Cho So, and Ye. 
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

#%% 
def check_symm(X, rtol = 1e-03, atol = 1e-03):
    return np.allclose(X, X.T, rtol=rtol, atol = atol)
    
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
    [mn, n] = A.shape
    m = mn//n #this SHOULD be an integer, since the A is created by stacking matrices on top of each other
    
    AV = A.dot(V) #mn by r matrix 
    AVs = np.vsplit(AV, m) #list of m arrays of size n by r
    
    VtAV_listmat = [V.transpose().dot(AVs[_]) for _ in range(m)] #rxr each 
    
    VtAV_listvec = [VtAV_listmat[_].flatten() for _ in range(m)] #r^2 each
    Av = np.asarray(VtAV_listvec) #m x r^2 matrix
    return Av
#%% 
def create_sym_map_mat(r2):
    #input: r^2
    # output: a matrix of all zeroes with ones in specific locations
    # size r(r+1)/2 x r^2. 
    r = np.int(np.sqrt(r2))
    numrows = np.int(r*(r+1)/2)
    numcols = r2
    M = np.zeros((numrows, numcols))
    
    #I simply don't know how to do this pythonically, so here's my awful C-ish for loop. 
    mat_dict = {} 
    col_tuples = [divmod(i, r) for i in range(numcols)] 
    num_repeats = 0
    for i in range(numcols):
        if(col_tuples[i][0] <= col_tuples[i][1]):
            row_pos = (col_tuples[i][0])*r + col_tuples[i][1] - num_repeats
            M[row_pos][i] = 1
            mat_dict[col_tuples[i]] = row_pos
        else:
            num_repeats+= 1 
            row_pos = mat_dict[(col_tuples[i][1], col_tuples[i][0])]
            M[row_pos][i] = 1
    
    return M
#%%       
def create_nullspace_AvCv(Av):
    #inputs: 
    # [Av:Cv] = m+1 by r^2 matrix, the i'th row being vec(V' A_i V), 
    # and the last one being vec(V' C V)
    
    #outputs:
    # Delta = r by r matrix obtained as follows: find a vector delta 
    # in the nullspace of [Av; Cv], where delta satisfies the  
    # condition that Delta is symmetric 
    
    # First, we reduce the dimension of our input matrix of which 
    # we intend to compute the nullspace. This dimension reduction 
    # ensures that the nullspace matrix we get is symmetric. 
    [mp1, r2] = Av.shape
    M = create_sym_map_mat(r2)
    compressed_Av = Av.dot(M.transpose()) #m+1 x r(r+1)/2
    
    #We then find A VECTOR in the nullspace of THIS compressed linear operator
    compressed_delta = sp.linalg.null_space(compressed_Av) #length r(r+1)/2
    compressed_delta_one = compressed_delta[:, 0]
    #Then we "de-compress" it
    delta = (np.transpose(M)).dot(compressed_delta_one) #r^2 
    
    # And finally, we reshape it into a matrix
    r = np.int(np.sqrt(r2))
    Delta = np.reshape(delta, (r, r))
    if (check_symm(Delta)):
        print("Delta is symmetric!")
    else:
        print("Error: Delta not symmetric")
    return Delta

#%% Construct the linear operator Cv = Vt * C * V
def create_Cv(C, V):
    #inputs: 
    # C = n by n objective matrix
    # V = n by r matrix

    #outputs: 
    # Cv = 1 x r^2 matrix obtained by ...
    # computing the product V^T C V (an r by r matrix),
    # then vectorizing it.  
    CV = C.dot(V) #mn by r matrix 
    VtCV = np.transpose(V).dot(CV)    
    VtCV_vec = VtCV.flatten()
    return VtCV_vec

#%%
def rank_reduction(X, A, C):
    #inputs:
    # X = n by n matrix, solution of given SDP
    # A = mn by n matrix, stack of m constraint matrices
    # C = n by n matrix, the objective matrix*
    #* Note that, in the algorithm, Section 2.2 of the monograph, 
    # we prove that we don't need V^T C V \bullet \Delta = 0 - this s
    # based on the assumption that X is a solution of the SDP. However, 
    # if this assumption isn't true, we can't use it! In our alg, we are 
    # actually perturbing the solution of the SDP to make it positive definite
    # so that we can apply Cholesky to it. Therefore, it's safer to just 
    # add this constraint. In any case, it's not much work to add it in.
    
    #outputs:
    # X = n by n matrix; solution to the SDP that has a rank <= that of the input

    # Cholesky factorize X to get V
    V = np.linalg.cholesky(X)
    
    # Construct some constants
    r = V.shape[1]
    I_r = np.eye(r)
    
    VtAV_vecmat = create_Av(A, V) #m x r^2
    VtCV_vec = create_Cv(C, V) #1 x r^2
    calAC = np.vstack((VtAV_vecmat, VtCV_vec)) #following monograph notation 
    
    count = 0
    # TODO error message if this isn't a valid op (#rows > #cols)
    Delta = create_nullspace_AvCv(calAC)
    while ( Delta.size != 0 and count < 100 and check_pd(X)!=0):
        # Find lambda_1 which is the max eigval of Delta
        lambda_1 = np.max(np.abs(np.linalg.eigvals(Delta)))
        # Choose alpha according to formula
        alpha = -1/lambda_1
        # Construct X = V(I + alpha * Delta)V'
        X = V.dot((I_r + alpha*Delta).dot(V.transpose()))
        # posdef'ize X
        X = make_pd(X)
        if (check_pd(X)):
            print("PD matrix before Cholesky")
        else:
            print("error  in Cholesky")
            return X
        # Compute Cholesky
        V = np.linalg.cholesky(X)
        # Create the matrices Vt*A[i]*V and Vt*C*V and vectorize and stack
        VtAV_vecmat = create_Av(A, V) #m x r^2
        VtCV_vec = create_Cv(C, V) #1 x r^2
        calAC = np.vstack((VtAV_vecmat, VtCV_vec)) #following monograph notation 
        # find symmetric Delta, matrix'ized nullspace(A_v)[:, 0]
        Delta = create_nullspace_AvCv(calAC) #TODO
        count = count + 1
        
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
            #TODO
            return
#%% 
def check_pd(X):
    #input: a symmetric matrix X
    #output: 1 or 0, pos_Def or not
    
    lambda_min = np.min(np.linalg.eigvals(X))
    if (lambda_min > 0):
        print("The min eigval is "+str(lambda_min))
        return 1
    else: return 0
       
#%%         
def make_pd(X):
    # input: A real, symmetric matrix X
    # output: Either the same matirx if it's PD, or a perturbed version which is. 
    X_evals, X_evecs = np.linalg.eig(X)
    lambda_min = np.min(X_evals)
    min_acceptable_val_for_pd = 1e-02
    eps = 1e-02
    
    if(lambda_min > min_acceptable_val_for_pd):
        print("The matrix is already positive definite")
        return X
    else:
        print("The matrix is NOT positive definite; adding appropriate perturbation")
        X_mod_evals = X_evals + (X_evals <= min_acceptable_val_for_pd)*(np.abs(lambda_min) + eps)
        X_mod = X_evecs.dot(np.diag(X_mod_evals).dot(X_evecs.transpose()))
        return X_mod
    
#%% 
def print_test_outputs(X_true, X_rred, sdp_instance):
    # First, check if the objectives match.
    C = sdp_instance.objective
    A = sdp_instance.constraints.A
    b = sdp_instance.constraints.b
    m = sdp_instance.num_constraints
    
    obj_val_diff = np.trace(C.dot(X_true)) - np.trace(C.dot(X_rred))
    print("The difference in objective values of the SDP solution and the rank reduced matrix is ", obj_val_diff)
    for i in range(m):
        print("Constraint "+str(i)+" violation: " +str(np.trace(A[i].dot(X_rred)) - b[i]))
    print("The eigenvalues of the rank reduced matrix are ", np.linalg.eigvals(X_rred))
    print("The eigenvalues of the original matrix are ", np.linalg.eigvals(X_true))
    return
#%%
if __name__ == '__main__':
    n = 5
    m = 3
    _testsdp_ = TestSDP(n, m)
    _testsdpsol_, _testsdpX_ = sdp(_testsdp_)
    A_cat = np.vstack(_testsdp_.constraints.A[i] for i in range(m))
    pdfied_X = make_pd(_testsdpX_.value)
    #_testsdp_.check_feasibility(pdfied_X)
    rred_X = rank_reduction(pdfied_X, A_cat, _testsdp_.objective)
    print_test_outputs(_testsdpX_.value, rred_X, _testsdp_)
    #TODO:
    # 1. Add make_pd inside rank-reduction alg, so that we can run it for more iters
    # 2. Do something about the numerical issue of eigvals being too tiny
    # 3. write the test for comparing rred_X to orig SDP sol : frob norm differences, 
    #    values of objective, constraints-feasibility, etc (w/ print statements)
                