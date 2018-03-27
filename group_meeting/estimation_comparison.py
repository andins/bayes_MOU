#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:35:54 2018

@author: andrea
"""
from MOU import MOU
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from make_rnd_connectivity import make_rnd_connectivity
from os import listdir
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns


# test estimation methods varying N
Ns = [10, 20, 50, 100]
repetitions = 100
file_name = "comparison_T500.npy"
if file_name in listdir():  # if file is in the directory load variables
    file = np.load(file_name)
    gen = file[0]
    lyap = file[1]
    mome = file[2]
else:  # else create the variables
    gen = dict()
    lyap = dict()
    mome = dict()
for N in Ns:
    if not(N in lyap.keys()):  # if the N has already been done skips
        gen[N] = dict()
        lyap[N] = dict()
        mome[N] = dict()
    for r in range(repetitions):
        if not(r in lyap[N].keys()):  # if the repetition has already been done skips
            try:  # if there is any error saves the results
                C = make_rnd_connectivity(N=N, density=0.2, connectivity_strength=0.5)
                generator = MOU(n_nodes=N, tau_x=1, mu=0.0, Sigma=None, C=C)
                ts = generator.simulate(T=500)
                gen[N][r] = generator
                model_l = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_l.fit(X=ts, method="lyapunov")
                lyap[N][r] = model_l
                model_m = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_m.fit(X=ts, method="moments")
                mome[N][r] = model_m
            except:  # here is the save
                np.save(file_name, (lyap, mome))
np.save(file_name, (gen, lyap, mome))

#%% plot
#rCl = np.zeros([len(Ns), repetitions])
#rCm = np.zeros([len(Ns), repetitions])
#for i, N in enumerate(Ns):
#    for r in range(repetitions):
#        rCl[i, r] = pearsonr(lyap[N][r].C.flatten(), gen[N][r].C.flatten())[0]
#        rCm[i, r] = pearsonr(mome[N][r].C.flatten(), gen[N][r].C.flatten())[0]
#plt.figure()
#plt.errorbar(Ns, rCl.mean(axis=1), rCl.std(axis=1))
#plt.errorbar(Ns, rCm.mean(axis=1), rCm.std(axis=1))


#%% test estimation methods varying T
N = 50
Ts = [500, 1000, 2000, 4000]
repetitions = 100
file_name = "comparison_N50.npy"
if file_name in listdir():  # if file is in the directory load variables
    file = np.load(file_name)
    gen = file[0]
    lyap = file[1]
    mome = file[2]
else:  # else create the variables
    gen = dict()
    lyap = dict()
    mome = dict()
for T in Ts:
    if not(T in lyap.keys()):  # if the N has already been done skips
        gen[T] = dict()
        lyap[T] = dict()
        mome[T] = dict()
    for r in range(repetitions):
        if not(r in lyap[T].keys()):  # if the repetition has already been done skips
            try:  # if there is any error saves the results
                C = make_rnd_connectivity(N=N, density=0.2, connectivity_strength=0.5)
                generator = MOU(n_nodes=N, tau_x=1, mu=0.0, Sigma=None, C=C)
                ts = generator.simulate(T=T)
                gen[T][r] = generator
                model_l = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_l.fit(X=ts, method="lyapunov")
                lyap[T][r] = model_l
                model_m = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_m.fit(X=ts, method="moments")
                mome[T][r] = model_m
            except:  # here is the save
                np.save(file_name, (lyap, mome))
np.save(file_name, (gen, lyap, mome))

#%% plot
rCl = np.zeros([len(Ts), repetitions])
rCm = np.zeros([len(Ts), repetitions])
for i, T in enumerate(Ts):
    for r in range(repetitions):
        rCl[i, r] = pearsonr(lyap[T][r].C.flatten(), gen[T][r].C.flatten())[0]
        rCm[i, r] = pearsonr(mome[T][r].C.flatten(), gen[T][r].C.flatten())[0]
plt.figure()
plt.errorbar(Ts, rCl.mean(axis=1), rCl.std(axis=1))
plt.errorbar(Ts, rCm.mean(axis=1), rCm.std(axis=1))

#%%  Estimate connectivity from data than simulate MOU for varying T and estimate C from simulated activity
N = 116
data = np.loadtxt('ROISignals_0025427_SE01.txt')
mask_AAL = np.array(loadmat('/home/andrea/Work/vicente/mask_EC_AAL.mat')['mask_EC'], dtype=bool)
model = MOU(n_nodes=N)
model.fit(X=data, SC_mask=mask_AAL)
Cemp = model.C
Ts = [500, 1000, 2000, 4000]
repetitions = 10
file_name = "comparison_N116_empC.npy"
if file_name in listdir():  # if file is in the directory load variables
    file = np.load(file_name)
    gen = file[0]
    lyap = file[1]
    mome = file[2]
else:  # else create the variables
    gen = dict()
    lyap = dict()
    mome = dict()
for T in Ts:
    if not(T in lyap.keys()):  # if the N has already been done skips
        gen[T] = dict()
        lyap[T] = dict()
        mome[T] = dict()
    for r in range(repetitions):
        if not(r in lyap[T].keys()):  # if the repetition has already been done skips
            try:  # if there is any error saves the results
                generator = MOU(n_nodes=N, tau_x=1, mu=0.0, Sigma=None, C=Cemp)
                ts = generator.simulate(T=T)
                gen[T][r] = generator
                model_l = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_l.fit(X=ts, method="lyapunov")
                lyap[T][r] = model_l
                model_m = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_m.fit(X=ts, method="moments")
                mome[T][r] = model_m
            except:  # here is the save
                np.save(file_name, (lyap, mome))
np.save(file_name, (gen, lyap, mome))

#%% plot
rCl = np.zeros([len(Ts), repetitions])
rCm = np.zeros([len(Ts), repetitions])
for i, T in enumerate(Ts):
    for r in range(repetitions):
        rCl[i, r] = pearsonr(lyap[T][r].C.flatten(), gen[T][r].C.flatten())[0]
        rCm[i, r] = pearsonr(mome[T][r].C.flatten(), gen[T][r].C.flatten())[0]
plt.figure()
plt.errorbar(Ts, rCl.mean(axis=1), rCl.std(axis=1))
plt.errorbar(Ts, rCm.mean(axis=1), rCm.std(axis=1))


#%%  Estimate connectivity from data than simulate MOU for varying strength of connection and estimate C from simulated activity
N = 116
T = 1000
data = np.loadtxt('ROISignals_0025427_SE01.txt')
mask_AAL = np.array(loadmat('/home/andrea/Work/vicente/mask_EC_AAL.mat')['mask_EC'], dtype=bool)
model = MOU(n_nodes=N)
model.fit(X=data, SC_mask=mask_AAL)
Cemp = model.C
Ss = [.8, 1.1, 1.3, 1.5]
repetitions = 10
file_name = "comparison_N116_CStrength.npy"
if file_name in listdir():  # if file is in the directory load variables
    file = np.load(file_name)
    gen = file[0]
    lyap = file[1]
    mome = file[2]
else:  # else create the variables
    gen = dict()
    lyap = dict()
    mome = dict()
for S in Ss:
    if not(S in lyap.keys()):  # if the N has already been done skips
        gen[S] = dict()
        lyap[S] = dict()
        mome[S] = dict()
    for r in range(repetitions):
        if not(r in lyap[S].keys()):  # if the repetition has already been done skips
            try:  # if there is any error saves the results
                generator = MOU(n_nodes=N, tau_x=1, mu=0.0, Sigma=None, C=Cemp*S)
                ts = generator.simulate(T=T)
                gen[S][r] = generator
                model_l = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_l.fit(X=ts, method="lyapunov")
                lyap[S][r] = model_l
                model_m = MOU(n_nodes=N, tau_x=1, mu=0.0)
                model_m.fit(X=ts, method="moments")
                mome[S][r] = model_m
            except:  # here is the save
                np.save(file_name, (lyap, mome))
np.save(file_name, (gen, lyap, mome))

#%% plot
rCl = np.zeros([len(Ss), repetitions])
rCm = np.zeros([len(Ss), repetitions])
for i, T in enumerate(Ts):
    for r in range(repetitions):
        rCl[i, r] = pearsonr(lyap[S][r].C.flatten(), gen[S][r].C.flatten())[0]
        rCm[i, r] = pearsonr(mome[S][r].C.flatten(), gen[S][r].C.flatten())[0]
plt.figure()
plt.errorbar(Ss, rCl.mean(axis=1), rCl.std(axis=1))
plt.errorbar(Ss, rCm.mean(axis=1), rCm.std(axis=1))

#%% estimate EC from movie dataset with moments and lyapunov method and compare subjects classification
# load dataset
ts_movie = np.load('/home/andrea/Work/vicente/data/movie/ts_emp.npy')    
# remove bad subjects: 1 11 19
ts_clean = np.delete(ts_movie, [0, 10, 18], 0)
# load Hagmann SC mask
movMask = np.load('/home/andrea/Work/vicente/mask_EC.npy')  # [roi, roi] the mask for existing EC connections
models_l = dict()
models_m = dict()
for sb in range(19):  # subjects loop
    models_l[sb] = dict()
    models_m[sb] = dict()
    for ss in range(5):  # sessions loop
        BOLD_ts = ts_clean[sb, ss, :, :].T
        models_l[sb][ss] = MOU(n_nodes=66)
        models_l[sb][ss].fit(X=BOLD_ts, method="lyapunov", SC_mask=movMask)
        models_m[sb][ss] = MOU(n_nodes=66)
        models_m[sb][ss].fit(X=BOLD_ts, method="moments", SC_mask=movMask)
file_name = "MOU_movie.npy"
np.save(file_name, (models_l, models_m))

# data matrix
Xl = np.zeros([19*5, np.sum(movMask.flatten())])
Xm = np.zeros([19*5, np.sum(movMask.flatten())])
i = 0
for sb in range(19):  # subjects loop
    for ss in range(5):  # sessions loop
        Xl[i, :] = models_l[sb][ss].C[movMask].flatten()
        Xm[i, :] = models_m[sb][ss].C[movMask].flatten()
        i += 1

#y = np.array([i for i in range(19) for sess_id in range(5)])  # labels
y = np.array([0 if sess_id<2 else 1 for i in range(19) for sess_id in range(5)])  # labels

clf = LogisticRegression(C=10000, penalty='l2', multi_class= 'multinomial', solver='lbfgs')

# corresponding pipeline: zscore and pca can be easily turned on or off
pipe_l = Pipeline([('zscore', StandardScaler()),
                         ('clf', clf)])
pipe_m = Pipeline([('zscore', StandardScaler()),
                         ('clf', clf)])
repetitions = 100  # number of times the train/test split is repeated
# shuffle splits for validation test accuracy
shS = ShuffleSplit(n_splits=repetitions, test_size=None, train_size=.8, random_state=0)

score_l = np.zeros([repetitions])
score_m = np.zeros([repetitions])

i = 0  # counter for repetitions
for train_idx, test_idx in shS.split(Xl):  # repetitions loop
    data_trainl = Xl[train_idx, :]
    data_trainm = Xm[train_idx, :]
    y_train = y[train_idx]
    data_testl = Xl[test_idx, :]
    data_testm = Xm[test_idx, :]
    y_test = y[test_idx]
    pipe_l.fit(data_trainl, y_train)
    pipe_m.fit(data_trainm, y_train)
    score_l[i] = pipe_l.score(data_testl, y_test)
    score_m[i] = pipe_m.score(data_testm, y_test)
    i+=1
        
# plot comparison as violin plots
fig, ax = plt.subplots()
sns.violinplot(data=[score_l, score_m], cut=0, orient='v', scale='width')
ax.set_xticklabels(['lyapunov', 'moments'])