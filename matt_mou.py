#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:56:57 2017

@author: andrea
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.linalg as spl
import scipy.stats as stt


def matt_mou(ts_emp, mask_EC=None, verbose=0, true_C=None, true_S=None):
    # optimzation steps and rate
    n_opt = 10000
    epsilon_EC = 0.0005
    epsilon_Sigma = 0.1
    regul_EC = 0  # 0.5
    regul_Sigma = 0  # 0.001
     
    print('regularization:', regul_EC, ';', regul_Sigma)
     
    N = 50 # number of ROIs
     
    n_tau = 3 # number of time shifts for FC_emp
    v_tau = np.arange(n_tau)
    i_tau_opt = 1 # time shift for optimization
     
    min_val_EC = 0. # minimal value for EC
    max_val_EC = 0.2 # maximal value for EC
    min_val_Sigma_diag = 0. # minimal value for Sigma
     
     
    # FC matrix (ts_emp matrix with stucture time x ROI index)
    n_T = ts_emp.shape[0]
    ts_emp -= np.outer(np.ones(n_T),ts_emp.mean(0))
    FC_emp = np.zeros([n_tau,N,N])
    for i_tau in range(n_tau):
        FC_emp[i_tau,:,:] = np.tensordot(ts_emp[0:n_T-n_tau,:],ts_emp[i_tau:n_T-n_tau+i_tau,:],axes=(0,0)) / float(n_T-n_tau-1)
     
    # autocovariance time constant
    log_ac = np.log(np.maximum(FC_emp.diagonal(axis1=1,axis2=2),1e-10))
    lin_reg = np.polyfit(np.repeat(v_tau,N),log_ac.reshape(-1),1)
    tau_x = -1./lin_reg[0]
    print('inverse of negative slope (time constant):', tau_x)
     
    # mask for existing connections for EC and Sigma
    mask_diag = np.eye(N,dtype=bool)
    if mask_EC is None:
        mask_EC = np.logical_not(mask_diag) # all possible connections except self
    mask_Sigma = np.eye(N,dtype=bool) # independent noise
#    mask_Sigma = np.ones([N,N],dtype=bool) # coloured noise
     
    # optimization
    print('*opt*')
    print('i tau opt:', i_tau_opt)
    tau = v_tau[i_tau_opt]
     
    # objective FC matrices (empirical)
    FC0_obj = FC_emp[0,:,:]
    FCtau_obj = FC_emp[i_tau_opt,:,:]
     
    coef_0 = np.sqrt(np.sum(FCtau_obj**2)) / (np.sqrt(np.sum(FC0_obj**2))+np.sqrt(np.sum(FCtau_obj**2)))
    coef_tau = 1. - coef_0
     
    # initial network parameters
    EC = np.zeros([N,N])
    Sigma = np.eye(N)  # initial noise
     
    # best distance between model and empirical data
    best_dist = 1e10
    best_Pearson = 0.
     
    # record model parameters and outputs
    dist_FC_hist = np.zeros([n_opt])*np.nan # FC error = matrix distance
    Pearson_FC_hist = np.zeros([n_opt])*np.nan # Pearson corr model/objective
    dist_C_hist = np.zeros([n_opt])*np.nan # FC error = matrix distance
    Pearson_C_hist = np.zeros([n_opt])*np.nan # Pearson corr model/objective
     
    stop_opt = False
    i_opt = 0
    while not stop_opt:
        # calculate Jacobian of dynamical system
        J = -np.eye(N)/tau_x + EC
             
        # calculate FC0 and FCtau for model
        FC0 = spl.solve_lyapunov(J,-Sigma)
        FCtau = np.dot(FC0,spl.expm(J.T*tau))
     
        # calculate error between model and empirical data for FC0 and FC_tau (matrix distance)
        err_FC0 = np.sqrt(np.sum((FC0-FC0_obj)**2))/np.sqrt(np.sum(FC0_obj**2))
        err_FCtau = np.sqrt(np.sum((FCtau-FCtau_obj)**2))/np.sqrt(np.sum(FCtau_obj**2))
        dist_FC_hist[i_opt] = 0.5*(err_FC0+err_FCtau)
         
        # calculate Pearson corr between model and empirical data for FC0 and FC_tau
        Pearson_FC_hist[i_opt] = 0.5*(stt.pearsonr(FC0.reshape(-1),FC0_obj.reshape(-1))[0]+stt.pearsonr(FCtau.reshape(-1),FCtau_obj.reshape(-1))[0])
     
        # calculate the error between true and estimated connectivity C if the true is provided (for debugging)
        if not(true_C is None):
            dist_C_hist[i_opt] =  np.sqrt(((true_C-EC)**2).sum()/(true_C**2).sum())
            Pearson_C_hist[i_opt] = stt.pearsonr(true_C.flatten(), EC.flatten())[0]
    
        # best fit given by best Pearson correlation coefficient for both FC0 and FCtau (better than matrix distance)
        if dist_FC_hist[i_opt]<best_dist:
            best_dist = dist_FC_hist[i_opt]
            best_Pearson = Pearson_FC_hist[i_opt]
            i_best = i_opt
            EC_best = np.array(EC)
            Sigma_best = np.array(Sigma)
            FC0_best = np.array(FC0)
            FCtau_best = np.array(FCtau)
        else:
            stop_opt = i_opt>100
     
        # Jacobian update with weighted FC updates depending on respective error
        Delta_FC0 = (FC0_obj-FC0)*coef_0
        Delta_FCtau = (FCtau_obj-FCtau)*coef_tau
        Delta_J = np.dot(np.linalg.pinv(FC0),Delta_FC0+np.dot(Delta_FCtau,spl.expm(-J.T*tau))).T/tau
     
        # update conectivity and noise
        EC[mask_EC] += epsilon_EC * (Delta_J - regul_EC*EC)[mask_EC]
        EC[mask_EC] = np.clip(EC[mask_EC],min_val_EC,max_val_EC)
     
        Sigma[mask_Sigma] += epsilon_Sigma * (-np.dot(J,Delta_FC0)-np.dot(Delta_FC0,J.T) - regul_Sigma)[mask_Sigma]
        Sigma[mask_diag] = np.maximum(Sigma[mask_diag],min_val_Sigma_diag)
     
        # check if end optimization: if FC error becomes too large
        if stop_opt or i_opt==n_opt-1:
            stop_opt = True
            print('stop at step', i_opt, 'with best dist', best_dist, ';best FC Pearson:', best_Pearson)
        else:
            if (i_opt)%20==0:
                print('opt step:', i_opt)
                print('current dist FC:', dist_FC_hist[i_opt], '; current Pearson FC:', Pearson_FC_hist[i_opt])
            i_opt += 1
     
        if verbose==2:
            plt.figure()
            plt.subplot(121)
            plt.plot(dist_FC_hist, label=r'$Q_{0/\tau}$ error')
            if not(true_C is None):
                plt.plot(dist_C_hist, label='C error')
                plt.legend()
            plt.ylabel('matrix error')
            plt.xlabel('iterations')
            plt.subplot(122)
            plt.plot(Pearson_FC_hist, label=r'$Q_{0/\tau}$ similarity')
            if not(true_C is None):
                plt.plot(Pearson_C_hist, label='C similarity')
                plt.legend()
            plt.ylabel('pearson correlation')
            plt.xlabel('iterations')
        
        return EC_best, Sigma_best
 