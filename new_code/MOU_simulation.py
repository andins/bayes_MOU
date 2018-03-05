#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:46:36 2017

@author: andrea
"""

import numpy as np

def MOU_sim(N=50, T=9000, dt=0.05, Sigma=None, mu=0, connectivity_strength=4., verbose=0):
    """
    N : number of network nodes
    T : duration of simulation
    dt : integration time step
    Sigma: the noise of the system.
    If a scalar s is given, a diagonal covariance is used with s on the diagonal.
    Otherwise defaults to a diagonal covariance with random values (uniform between 0.5 and 1)
    There is code to use colored noise (off-diagonal terms) but now is not in use.
    connectivity_strength: scales the values of the weigths as connectivity_strength * N / sum(C).
    The value of the connection has to be bounded depending on the dimensionality N in order to avoid
    explosion of the dynamics. As an indication for N=50 weigths should be <1.9.
    """
    ##################
    # model parameters
    
    T0 = 100. # initialization time for network dynamics
    
    n_sampl = int(1./dt) # sampling to get 1 point every second
    tau_x = 1. # time constant of dynamical system
    
    
    # input noise matrix (diagonal only so far)
    coef_Sigma_diag = 1.
    coef_Sigma_offdiag = 0  # 0.2
    
    
    # activation function Phi
    act_mode = 'lin'
    if verbose>0:
        print('act_mode:', act_mode)
    
    if act_mode=='lin':
    	# linear
    	a = 0.1
    	param_Phi = [a]
    	def Phi(x,param):
    		a = param[0]
    		return x*a
    elif act_mode=='exp':
    	# exponential
    	a = 0.2
    	b = -2.
    	param_Phi = [a,b]
    	def Phi(x,param):
    		a,b = param
    		return np.exp(a*x+b)
    else:
    	raise ValueError('wrong act mode!')
    
    
    # artificial network with connectivity ollowing structure in SC matrix
    C_orig = np.random.rand(N,N)
    C_orig[np.random.rand(N,N)>0.2] = 0
    C_orig[np.eye(N, dtype=bool)] = 0
#    C_orig *= np.random.randn(N, N) + connectivity_strength  # gaussian weights
    C_orig *= connectivity_strength * N / C_orig.sum()
    
    
    # input mean
    I_orig = mu  # np.random.rand(N) can be used to have inhomogeneous inputs
    
    # this part will be used for colored noise (coef to 0 at the moment #######
    # input noise matrix from mixing Gaussian White noise
    Sigma_mixing = coef_Sigma_offdiag * (2*np.random.rand(N,N)-1)
    Sigma_mixing[np.random.rand(N,N)>0.1] = 0
    ###########################################################################
    if Sigma is None:
        Sigma_mixing[np.eye(N,dtype=bool)] = coef_Sigma_diag * (0.5+0.5*np.random.rand(N))
    elif np.isscalar(Sigma):
        Sigma_mixing[np.eye(N,dtype=bool)] = np.sqrt(Sigma)
    Sigma_orig = np.dot(Sigma_mixing,Sigma_mixing.T)
    
    
    # numerical simulations
    n_T = int(np.ceil(T/dt))
    n_T0 = int(T0/dt)
    ts_emp = np.zeros([n_T,N]) # to save results
    # initialization
    t_span = np.arange(n_T0+n_T,dtype=int)
    x_tmp = np.random.rand(N)
    u_tmp = np.zeros([N])
    noise = np.random.normal(size=[n_T0+n_T,N],scale=(dt**0.5))
    for t in t_span:
    	u_tmp = np.dot(C_orig,x_tmp) + I_orig
    	x_tmp += dt * ( -x_tmp/tau_x + Phi(u_tmp,param_Phi) ) + np.dot(Sigma_mixing,noise[t,:])
    	if t>n_T0:
    		ts_emp[t-n_T0,:] = x_tmp
    
    # return data
    return ts_emp[::n_sampl,:], C_orig, Sigma_orig


X, C, S = MOU_sim(N=10, Sigma=None, mu=0, T=300, connectivity_strength=1.5)