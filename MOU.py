"""
Created on Sun Oct 1 15:50:00 2017

Simulation of Multivariate Ornstein-Uhlenbeck process

@author: andrea insabato
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.linalg as spl
import scipy.stats as stt


######################################################################

def MOU_sim(M=2, duration=300, regr_matrix=1):
    # this simulate the DS with Euler integration
    if np.isscalar(regr_matrix):  # if is not provided use a random one
        tau = 1
        C = np.random.randn(M,M) * 0.2
    else:
        C = regr_matrix.copy()
        C[np.eye(M, dtype=bool)] = 0
        tau = 1  # -1/regr_matrix[np.eye(M, dtype=bool)]
    e = np.ones([1, M]) * 0
    sigma = 0.6
    ini = np.random.randn(M)
    dt = 0.05
    steps = int(duration/dt)
    X = np.zeros([M, steps])
    X[:, 0] = ini
    noise = np.random.randn(M, steps) * np.sqrt(sigma) * np.sqrt(dt)
    for i in range(1, steps):
#        X[:, i] = X[:, i-1] + dt * ( np.dot(-lam, X[:, i-1]) + e ) + noise[:, i-1]    
        X[:, i] = X[:, i-1] + dt * ( -X[:, i-1]/tau + np.dot(C, X[:, i-1]) + e ) + noise[:, i-1]
    return X, C


def MOU_estim_Lyapunov(X, mask_EC, verbose=0, true_C=None, true_S=None):
    # Lyapunov optimization of MOU
    
    N = X.shape[0]  # number of ROIs
    T = X.shape[1]  # number of TRs of the recording
    # time shifts for FC: 0, 1 and 2 TR
    v_tau = np.arange(3,dtype=float)
    n_tau = v_tau.size


    FC_emp = np.zeros([n_tau,N,N])
    X -= np.outer(X.mean(axis=1), np.ones(T))  # center the time series
    for i_tau in range(n_tau):
        # FIXME: sobstituted here tensordot() for dot() since there is only one session
        FC_emp[i_tau,:,:] = np.dot(X[:, 0:T-n_tau+1], X[:,i_tau:T-n_tau+1+i_tau].T) / float(T-n_tau)

    # FC_emp has to be between 0 and 1 otherwise the connectivity is to high and the system explodes
    FC_emp *= 0.5/FC_emp[0,:,:].mean()
    if verbose>0:
        print('max FC value (most of the distribution should be between 0 and 1):', FC_emp.mean())
        
    # time constant for BOLD autocovariances
    slopes = np.zeros([N])
    if verbose==2:
        plt.figure()
    for i in range(N):
        ac_tmp = np.maximum(FC_emp[:, i, i], 1e-10)  # autocovariance for time shifts in v_tau; with lower bound to avoid negative values (cf. log)
        slopes[i] = np.polyfit(v_tau, np.log(ac_tmp), 1)[0]  # slope of autocovariance for ROI i
        if verbose==2:
            plt.plot(v_tau, np.log(ac_tmp))

    tau_x = -1. / slopes.mean()  # inverse of negative slope of autocovariance

    # diagonal mask for input noise matrix (here, no input cross-correlation)
    mask_Sigma = np.eye(N,dtype=bool)

    w_C = 1. # weight reference to set max in optimization (to increase if too many saturated estimated weights)
    
    # optimzation rates (to avoid explosion of activity, Sigma is tuned quicker)
    epsilon_EC = 0.0005
    epsilon_Sigma = 0.05
    
    min_val_EC = 0. # minimal value for tuned EC elements
    max_val_EC = 1. # maximal value for tuned EC elements
    min_val_Sigma = 0. # minimal value for tuned Sigma elements

    # time shifts for FC: 0, 1 and 2 TR
    v_tau = np.arange(3,dtype=float)
    n_tau = v_tau.size
    
    i_tau = 1 # time shift for optimization (in TR units; can be 1 or 2)
    tau = v_tau[i_tau]
    if verbose>0:
        print('opt with time shift', tau, 'TR')
    
    # objective FC matrices (empirical)
    FC0_obj = FC_emp[0,:,:]
    FCtau_obj = FC_emp[i_tau,:,:]

    # initializing
    EC_mod = np.zeros([N,N])
    Sigma_mod = np.zeros([N,N])
    FC0_mod = np.zeros([N,N])
    FCtau_mod = np.zeros([N,N])
    
    # initial EC
    EC = np.zeros([N,N]) # initial connectivity
    Sigma = np.eye(N)  # initial noise
    # record best fit (matrix distance between model and empirical FC)
    best_dist = 1e10
    
    dist_FC_tmp = list()  # np.zeros(101)
    Pearson_FC_tmp = list() # np.zeros(101)
    dist_C = list()  # np.zeros(101)
    Pearson_C = list()  # np.zeros(101)
    dist_S = list()
    Pearson_S = list()
    stop_opt = False
    i_opt = 0
    while not stop_opt:

        # calculate Jacobian of dynamical system
        J = -np.eye(N)/tau_x + EC
        
        # calculate FC0 and FCtau for model
        FC0 = spl.solve_lyapunov(J,-Sigma)
        FCtau = np.dot(FC0,spl.expm(J.T*tau))
        # matrices of model error
        Delta_FC0 = FC0_obj-FC0
        Delta_FCtau = FCtau_obj-FCtau
        
        # calculate error between model and empirical data for FC0 and FC_tau (matrix distance)
        dist_FC_tmp.append( 0.5*(np.sqrt((Delta_FC0**2).sum()/(FC0_obj**2).sum())+np.sqrt((Delta_FCtau**2).sum()/(FCtau_obj**2).sum())) )
        
        # calculate Pearson correlation between model and empirical data for FC0 and FC_tau
        Pearson_FC_tmp.append( 0.5*(stt.pearsonr(FC0.reshape(-1),FC0_obj.reshape(-1))[0]+stt.pearsonr(FCtau.reshape(-1),FCtau_obj.reshape(-1))[0]) )
        
        # calculate the error between true and estimated connectivity C if the true is provided (for debugging)
        if not(true_C is None):
            dist_C.append( np.sqrt(((true_C-EC)**2).sum()/(true_C**2).sum()) )
            Pearson_C.append( stt.pearsonr(true_C.flatten(), EC.flatten())[0] )
            
        # calculate the error between true and estimated Sigma if the true is provided (for debugging)
        if not(true_S is None):
            dist_S.append( np.sqrt(((true_S-Sigma)**2).sum()/(true_S**2).sum()) )
            Pearson_S.append( stt.pearsonr(true_S.flatten(), Sigma.flatten())[0] )
        
        # record best model parameters
        if dist_FC_tmp[i_opt]<best_dist:
            best_dist = dist_FC_tmp[i_opt]
            best_Pearson = Pearson_FC_tmp[i_opt]
            i_best = i_opt
            EC_mod_tmp = np.array(EC)
            Sigma_mod_tmp = np.array(Sigma)
            FC0_mod_tmp = np.array(FC0)
            FCtau_mod_tmp = np.array(FCtau)
        else:
            stop_opt = i_opt>=100
        
        # Jacobian update
        Delta_J = np.dot(np.linalg.pinv(FC0),Delta_FC0+np.dot(Delta_FCtau,spl.expm(-J.T*tau))).T/tau
        
        # update EC (recurrent connectivity)
        EC[mask_EC] += epsilon_EC * Delta_J[mask_EC]
        EC[mask_EC] = np.clip(EC[mask_EC],min_val_EC,max_val_EC)
        
        # update Sigma (input variances)
        Delta_Sigma = -np.dot(J,Delta_FC0)-np.dot(Delta_FC0,J.T)
        Sigma[mask_Sigma] += epsilon_Sigma * Delta_Sigma[mask_Sigma]
        Sigma[mask_Sigma] = np.maximum(Sigma[mask_Sigma],min_val_Sigma)
        
        # check for stop
        if not stop_opt:
            if verbose>0 and (i_opt)%50==0:
                print('opt step:', i_opt)
                print('dist FC:', dist_FC_tmp[i_opt], '; Pearson FC:', Pearson_FC_tmp[i_opt])
            i_opt += 1
        else:
            if verbose>0:
                print('stop at step', i_opt, 'with best FC dist:', best_dist, '; best FC Pearson:', best_Pearson)

    if verbose==2:
        plt.figure()
        plt.subplot(121)
        plt.plot(dist_FC_tmp, label=r'$Q_{0/\tau}$ error')
        if not(true_C is None):
            plt.plot(dist_C, label='C error')
            plt.legend()
        if not(true_S is None):
            plt.plot(dist_S, label=r'$\Sigma$ error')
            plt.legend()
        plt.ylabel('matrix error')
        plt.xlabel('iterations')
        plt.subplot(122)
        plt.plot(Pearson_FC_tmp, label=r'$Q_{0/\tau}$ similarity')
        if not(true_C is None):
            plt.plot(Pearson_C, label='C similarity')
            plt.legend()
        if not(true_S is None):
            plt.plot(Pearson_S, label=r'$\Sigma$ similarity')
            plt.legend()
        plt.ylabel('pearson correlation')
        plt.xlabel('iterations')
        
    EC_mod = EC_mod_tmp
    Sigma_mod = Sigma_mod_tmp
    FC0_mod = FC0_mod_tmp
    FCtau_mod = FCtau_mod_tmp

    return EC_mod, Sigma_mod, FC0_mod, FCtau_mod


def matrix_norm_dist(M1, M2):
    M1 = np.double(M1) 
    M2 = np.double(M2)
    d = np.sum((M1-M2)**2) / np.sum(M2**2)
    return d

############################################################################

