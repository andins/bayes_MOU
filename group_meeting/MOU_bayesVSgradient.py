#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 08:00:51 2018

@author: andrea
"""
import sys
sys.path.append("/home/andrea/Work/MOU")
from MOU import MOU, make_rnd_connectivity, classfy
from scipy.io import loadmat
import numpy as np
from scipy.stats import pearsonr
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
import seaborn as sns

#%% Univariate¶
sns.set_context('talk')

model = MOU(n_nodes=1, tau_x=50, mu=0.0, Sigma=1e-18)
ts = model.simulate(T=500)
plt.figure()
plt.plot(ts)
plt.xlabel('time')
plt.show()

#%% Mutlivariate with random connectivity matrix
sns.set_style('dark')
C = make_rnd_connectivity(N=10, density=0.2)
model = MOU(n_nodes=10, tau_x=1, mu=0.0, Sigma=None, C=C, random_state=0)
ts = model.simulate(T=500, random_state=0)
plt.figure()
plt.subplot(1,3,1)
plt.imshow(C, cmap='PiYG')
plt.xlabel('source')
plt.ylabel('target')
plt.colorbar()
plt.subplot(1,2,2)
plt.plot(ts)
plt.xlabel('time')
plt.show()

#%% Estimation accuracy varying M (with synthetic data)
sns.set_style('darkgrid')
Ms = [10, 20, 50, 100]
repetitions = 10  # 10 repetitions already gives a good idea of variability with short execution time
rCl = np.zeros([len(Ms), repetitions])
rCm = np.zeros([len(Ms), repetitions])
rSl = np.zeros([len(Ms), repetitions])
rSm = np.zeros([len(Ms), repetitions])
for i, M in enumerate(Ms):
    for r in range(repetitions):
        # create generative connectivity matrix
        C = make_rnd_connectivity(N=M, density=0.2, connectivity_strength=0.5)
        # simulate time series
        gen_model = MOU(n_nodes=M, tau_x=1, mu=0.0, Sigma=None, C=C)
        Sigma = gen_model.Sigma
        ts = gen_model.simulate(T=500)
        # instantiate the model with the two estimation methods and fit them
        model_l = MOU(n_nodes=M, tau_x=1, mu=0.0).fit(X=ts, method="lyapunov")
        model_m = MOU(n_nodes=M, tau_x=1, mu=0.0).fit(X=ts, method="moments")
        # compute estimation accuracies
        rCl[i, r] = pearsonr(model_l.C.flatten(), C.flatten())[0]
        rCm[i, r] = pearsonr(model_m.C.flatten(), C.flatten())[0]
        rSl[i, r] = pearsonr(model_l.Sigma[np.eye(M, dtype=bool)], Sigma[np.eye(M, dtype=bool)])[0]
        rSm[i, r] = pearsonr(model_m.Sigma[np.eye(M, dtype=bool)], Sigma[np.eye(M, dtype=bool)])[0]
plt.figure()
plt.subplot(1,2,1)
plt.errorbar(Ms, rCl.mean(axis=1), rCl.std(axis=1), label='lyapunov')
plt.errorbar(Ms, rCm.mean(axis=1), rCm.std(axis=1), label='moments')
plt.xlabel('# nodes')
plt.ylabel('estimation accuracy')
plt.title('C estimation')
plt.subplot(1,2,2)
plt.errorbar(Ms, rSl.mean(axis=1), rSl.std(axis=1), label='lyapunov')
plt.errorbar(Ms, rSm.mean(axis=1), rSm.std(axis=1), label='moments')
plt.xlabel('# nodes')
plt.title(r'$\Sigma$ estimation')
plt.legend()
plt.show()

#%% Estimation accuracy varying T (fixed M)
M = 50
Ts = [500, 1000, 2000, 4000]
repetitions = 10
rCl = np.zeros([len(Ts), repetitions])
rCm = np.zeros([len(Ts), repetitions])
for i, T in enumerate(Ts):
    for r in range(repetitions):
        # create generative connectivity matrix
        C = make_rnd_connectivity(N=M, density=0.2, connectivity_strength=0.5)
        # simulate time series
        gen_model = MOU(n_nodes=M, tau_x=1, mu=0.0, Sigma=None, C=C)
        Sigma = gen_model.Sigma
        ts = gen_model.simulate(T=T)
        # instantiate the model with the two estimation methods and fit them
        model_l = MOU(n_nodes=M, tau_x=1, mu=0.0).fit(X=ts, method="lyapunov")
        model_m = MOU(n_nodes=M, tau_x=1, mu=0.0).fit(X=ts, method="moments")
        # compute estimation accuracies
        rCl[i, r] = pearsonr(model_l.C.flatten(), C.flatten())[0]
        rCm[i, r] = pearsonr(model_m.C.flatten(), C.flatten())[0]
        rSl[i, r] = pearsonr(model_l.Sigma[np.eye(M, dtype=bool)], Sigma[np.eye(M, dtype=bool)])[0]
        rSm[i, r] = pearsonr(model_m.Sigma[np.eye(M, dtype=bool)], Sigma[np.eye(M, dtype=bool)])[0]
plt.figure()
plt.subplot(1,2,1)
plt.errorbar(Ts, rCl.mean(axis=1), rCl.std(axis=1), label='lyapunov')
plt.errorbar(Ts, rCm.mean(axis=1), rCm.std(axis=1), label='moments')
plt.xlabel('# time samples')
plt.ylabel('estimation accuracy')
plt.title('C estimation')
plt.subplot(1,2,2)
plt.errorbar(Ts, rSl.mean(axis=1), rSl.std(axis=1), label='lyapunov')
plt.errorbar(Ts, rSm.mean(axis=1), rSm.std(axis=1), label='moments')
plt.xlabel('# time samples')
plt.title(r'$\Sigma$ estimation')
plt.legend()
plt.show()

#%% Connectivity estimated from data
# load BOLD time series
data = np.loadtxt('ROISignals_0025427_SE01.txt')
# load structural connectivity mask
mask_AAL = np.array(loadmat('/home/andrea/Work/vicente/mask_EC_AAL.mat')['mask_EC'], dtype=bool)
model = MOU(n_nodes=116)
model.fit(X=data, SC_mask=mask_AAL)
Cemp = model.C
sns.set_style('dark')
fig, ax = plt.subplots(nrows=1, ncols=2)
mapcol = ax[0].imshow(Cemp, cmap='PiYG')
ax[0].set_xlabel('source')
ax[0].set_ylabel('target')
plt.colorbar(mappable=mapcol, ax=ax[0], shrink=.7)

# Estimation varying T (with empirical connectivity)
M = 116
Ts = [500, 1000, 2000, 4000]
repetitions = 10
rCl = np.zeros([len(Ts), repetitions])
rCm = np.zeros([len(Ts), repetitions])
for i, T in enumerate(Ts):
    for r in range(repetitions):
        # simulate model with connectivity Cemp (calculated in previous cell)
        ts = MOU(n_nodes=M, tau_x=1, mu=0.0, Sigma=None, C=Cemp).simulate(T=T)
        # instantiate the model with the two estimation methods and fit them
        model_l = MOU(n_nodes=M, tau_x=1, mu=0.0).fit(X=ts, method="lyapunov")
        model_m = MOU(n_nodes=M, tau_x=1, mu=0.0).fit(X=ts, method="moments")
        # compute estimation accuracies
        rCl[i, r] = pearsonr(model_l.C.flatten(), Cemp.flatten())[0]
        rCm[i, r] = pearsonr(model_m.C.flatten(), Cemp.flatten())[0]

sns.set_style('darkgrid')
ax[1].errorbar(Ts, rCl.mean(axis=1), rCl.std(axis=1), label='lyapunov')
ax[1].errorbar(Ts, rCm.mean(axis=1), rCm.std(axis=1), label='moments')
ax[1].set_xlabel('# time samples')
ax[1].set_ylabel('estimation accuracy')
ax[1].legend()
plt.show()

#%% Application to cognitive state classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# estimating the connectivity take time, you can skip this step and instead directly load the data matrices for classification
estimate_connectivity = False
file_name = 'EC_datamatrix_movie.npy'

# load Hagmann SC mask
movMask = np.load('/home/andrea/Work/vicente/mask_EC.npy')  # [roi, roi] the mask for existing EC connections
# load dataset
ts_movie = np.load('/home/andrea/Work/vicente/data/movie/ts_emp.npy')    
# remove bad subjects: 1 11 19
ts_clean = np.delete(ts_movie, [1, 11, 19], 0)

if estimate_connectivity:
    ######## Estimate connectivity from data ############
    models_l = dict()
    models_m = dict()
    # estimate connectivity
    for sb in range(19):  # subjects loop
        models_l[sb] = dict()
        models_m[sb] = dict()
        for ss in range(5):  # sessions loop
            BOLD_ts = ts_clean[sb, ss, :, :].T
            models_l[sb][ss] = MOU(n_nodes=66)
            models_l[sb][ss].fit(X=BOLD_ts, method="lyapunov", SC_mask=movMask)
            models_m[sb][ss] = MOU(n_nodes=66)
            models_m[sb][ss].fit(X=BOLD_ts, method="moments", SC_mask=movMask)
    #####################################################


    ############ data matrix for classification   ##############
    Xl = np.zeros([19*5, np.sum(movMask.flatten())])
    Xm = np.zeros([19*5, np.sum(movMask.flatten())])
    i = 0
    for sb in range(19):  # subjects loop
        for ss in range(5):  # sessions loop
            Xl[i, :] = models_l[sb][ss].C[movMask].flatten()
            Xm[i, :] = models_m[sb][ss].C[movMask].flatten()
            i += 1
    #####################################################
else:
    Xl, Xm = np.load(file_name)

# estimate FC
FC = np.zeros([19, 5, 66, 66])
for sb in range(19):  # subjects loop
    for ss in range(2):  # sessions loop
        BOLD_ts = ts_clean[sb, ss, :, :]
        FC[sb, ss, :, :] = np.corrcoef(BOLD_ts)
# data matrix of FC with SC mask
Xfc = np.zeros([19*5, np.sum(movMask.flatten())])
i = 0
for sb in range(19):  # subjects loop
    for ss in range(5):  # sessions loop
        Xfc[i, :] = FC[sb, ss, movMask]
        i += 1
# labels
y = np.array([0 if sess_id<2 else 1 for i in range(19) for sess_id in range(5)])  # condition labels
#y = np.array([i for i in range(19) for sess_id in range(5)])  # subject labels

# calculate test-set classification accuracy
score_l = classfy(X=Xl, y=y)  # lyapunov
score_m = classfy(X=Xm, y=y)  # moments
score_fc = classfy(X=Xfc, y=y)  # moments

# plot comparison as violin plots
fig, ax = plt.subplots()
sns.violinplot(data=[score_l, score_m, score_fc], cut=0, orient='v', scale='width')
ax.set_xticklabels(['lyapunov', 'moments', 'Pearson FC'])
plt.ylabel('classification accuracy')
plt.show()

#%% why does Bayes MAP fail

sns.set_style('darkgrid')
Ms = [10, 20, 50, 100]
repetitions = 10  # 10 repetitions already gives a good idea of variability with short execution time
rCm = np.zeros([len(Ms), repetitions])
rL = np.zeros([len(Ms), repetitions])
rQ0 = np.zeros([len(Ms), repetitions])
rQ1 = np.zeros([len(Ms), repetitions])
rPrec = np.zeros([len(Ms), repetitions])
C_im = np.zeros([len(Ms), repetitions])

for i, M in enumerate(Ms):
    for r in range(repetitions):
        # create generative connectivity matrix
        C = make_rnd_connectivity(N=M, density=0.2, connectivity_strength=0.5)
        # theoretical Lambda
        J = C.copy()
        np.fill_diagonal(J, -1)
        LAM_gen = expm(J)  # maybe add diagonal
        # generative model
        gen_model = MOU(n_nodes=M, tau_x=1, mu=0.0, Sigma=None, C=C)
        # theoretical covariances
        Q0_gen = gen_model.model_covariance(tau=0)
        Q1_gen = gen_model.model_covariance(tau=1)
        prec_gen = np.linalg.inv(Q0_gen)
        # simulate time series
        ts = gen_model.simulate(T=500)
        # instantiate the estimation model and fit it
        #model_m = MOU(n_nodes=M, tau_x=1, mu=0.0).fit(X=ts, method="moments")
        # estimate $\Lambda=(Q^0)^{-1} Q^{\tau}$, Q0 and Q1
        ts_zeromean = ts - np.outer(np.ones(500), ts.mean(0))  # subtract mean
        Q0_hat = np.dot(ts_zeromean.T, ts_zeromean) / 498
        Q1_hat = np.dot(ts_zeromean[:-1, :].T, ts_zeromean[1:, :]) / 498
        prec_hat = np.linalg.inv(Q0_hat)
        LAM_hat = np.dot(prec_hat, Q1_hat)
        C_hat = logm(LAM_hat).T
        np.fill_diagonal(C_hat, 0)
        # norm of imaginary part of C_hat
        C_im[i, r] = np.linalg.norm(np.imag(C_hat))
        C_hat = np.real(C_hat)  # cast to real
        # compute estimation accuracies
        rCm[i, r] = pearsonr(C_hat.flatten(), C.flatten())[0]
        rL[i, r] = pearsonr(LAM_gen.flatten(), LAM_hat.flatten())[0]
        rQ0[i, r] = pearsonr(Q0_gen.flatten(), Q0_hat.flatten())[0]
        rQ1[i, r] = pearsonr(Q1_gen.flatten(), Q1_hat.flatten())[0]
        rPrec[i, r] = pearsonr(prec_gen.flatten(), prec_hat.flatten())[0]
plt.figure()
plt.subplot(1, 2, 1)
plt.errorbar(Ms, rCm.mean(axis=1), rCm.std(axis=1), label='C')
plt.errorbar(Ms, rL.mean(axis=1), rL.std(axis=1), label=r'$\Lambda$')
plt.errorbar(Ms, rQ0.mean(axis=1), rQ0.std(axis=1), label=r'$Q^0$')
plt.errorbar(Ms, rQ1.mean(axis=1), rQ1.std(axis=1), label=r'$Q^1$')
plt.errorbar(Ms, rPrec.mean(axis=1), rPrec.std(axis=1), label=r'$(Q^0)^{-1}$')
plt.xlabel('# nodes')
plt.ylabel('estimation accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.errorbar(Ms, C_im.mean(axis=1), C_im.std(axis=1), label='C')
plt.xlabel('# nodes')
plt.ylabel('norm of imaginary part of C')
plt.show()