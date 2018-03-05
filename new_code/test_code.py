#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:04:18 2017

@author: andrea
"""
#%%
from MOU_simulation import MOU_sim
from MOU_estimation import MOU_Lyapunov
from MOU_bayes import map_MOU
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import logm
from scipy.stats import pearsonr
sns.set_context('notebook')
X, C, S = MOU_sim(N=50, Sigma=None, mu=0, T=1000, connectivity_strength=8.)
EC, Sl, taul = MOU_Lyapunov(X, true_C=C, true_S=S, verbose=2)
Cb, Sb, taub = map_MOU(X.T, verbose=2)
plt.figure()
plt.scatter(C, Cb)
plt.xlabel('true weights')
plt.ylabel('estimated weigths')
plt.text(plt.xlim()[0]+.05, plt.ylim()[1]-.05, 
         r'$\rho$: ' + str(pearsonr(Cb.flatten(), C.flatten())[0]))

#%% try VI with Edward

import edward as ed
import tensorflow as tf
from edward.models import Normal, InverseGamma, PointMass, Uniform
# simulate data
d = 50
T = 300
X, C, S = MOU_sim(N=d, Sigma=None, mu=0, T=T, connectivity_strength=8.)

# the model
mu = Uniform(low=tf.ones([d])*-1, high=tf.ones([d])*1)  # Normal(loc=tf.zeros([d]), scale=10.*tf.ones([d]))
beta = Uniform(low=tf.ones([d, d])*-20, high=tf.ones([d, d])*20)  # Normal(loc=tf.zeros([d, d]), scale=2.*tf.ones([d,d]))

noise_proc = tf.constant(0.1)  # InverseGamma(alpha=1.0, beta=1.0)
noise_obs = tf.constant(0.1)  # InverseGamma(alpha=1.0, beta=1.0)

x = [0] * T
x[0] = Normal(loc=mu, scale=10.*tf.ones([d]))
for n in range(1, T):
    x[n] = Normal(loc=mu + tf.tensordot(beta, x[n-1], axes=[[1],[0]]),
                  scale=noise_proc*tf.ones([d]))
## map inference
#print("setting up distributions")
#qmu = PointMass(params=tf.Variable(tf.zeros([d])))
#qbeta = PointMass(params=tf.Variable(tf.zeros([d,d])))
#print("constructing inference object")
#inference = ed.MAP({beta: qbeta, mu: qmu}, data={xt: xt_true for xt, xt_true in zip(x, X)})
#print("running inference")
#inference.run()
#
#print("parameter estimates:")
#print("mu: ", qmu.value().eval())
#print("beta:\n", qbeta.value().eval())

Cb, Sb, taub = map_MOU(X.T, verbose=2)
Cb[np.eye(d, dtype=bool)] = taub

# VI
print("setting up variational distributions")
qmu = Normal(loc=tf.Variable(tf.random_normal([d])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([d]))))
qbeta = Normal(loc=tf.Variable(tf.random_normal([d, d])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([d,d]))))
print("constructing inference object")
%time inference_vb = ed.KLqp({beta: qbeta, mu: qmu}, data={xt: xt_true for xt, xt_true in zip(x, X)})
print("running VB inference")
inference_vb.run()

Cvb = qbeta.mean().eval()

pp = qbeta.cdf(0.).eval()
Cvb_filt = Cvb.copy()
Cvb_filt[pp<0.05] = 0
off_diag_mask = [np.logical_not(np.eye(d, dtype=bool))]
print(pearsonr(C[off_diag_mask], Cvb[off_diag_mask]))
print(pearsonr(C[off_diag_mask], Cvb_filt[off_diag_mask]))
plt.figure()
plt.subplot(121)
plt.scatter(C[off_diag_mask], Cvb[off_diag_mask])
plt.subplot(122)
plt.scatter(C[off_diag_mask], Cvb_filt[off_diag_mask])
plt.figure()
plt.subplot(131)
sns.heatmap(C)
plt.subplot(132)
sns.heatmap(Cvb)
plt.subplot(133)
sns.heatmap(pp<0.05)
#%% estimation varying T
Ts = [200, 400, 800, 1600, 3200]
m = 50  # dimensionality
repetitions = 5
pearsonL = np.zeros([np.size(Ts), repetitions])
pearsonB = np.zeros([np.size(Ts), repetitions])
pearsonM = np.zeros([np.size(Ts), repetitions])
pears_sigmaL = np.zeros([np.size(Ts), repetitions])
pears_sigmaB = np.zeros([np.size(Ts), repetitions])

for i,t in enumerate(Ts):
    for r in range(repetitions):
        conn_strength = 1.8/m*(m**2*.2-m)/2  # to get approx connectivity strength in the same range when varying dimensionality
        X, C, S = MOU_sim(N=m, Sigma=None, T=t, connectivity_strength=conn_strength)
        Cl, Sl, taul = MOU_Lyapunov(X, verbose=0)
        Cb, Sb, taub = map_MOU(X.T, verbose=0)
        Cm = Cb.copy()
        Cm[Cm<0] = 0
        pearsonL[i, r] = pearsonr(Cl.flatten(), C.flatten())[0]
        pearsonB[i, r] = pearsonr(Cb.flatten(), C.flatten())[0]
        pearsonM[i, r] = pearsonr(Cm.flatten(), C.flatten())[0]
        pears_sigmaL[i, r] = pearsonr(Sl.diagonal(), S.diagonal())[0]
        pears_sigmaB[i, r] = pearsonr(Sb.diagonal(), S.diagonal())[0]

plt.figure()
plt.subplot(121)
plt.errorbar(Ts, pearsonB.mean(axis=1), pearsonB.std(axis=1), label='MAP')
plt.errorbar(Ts, pearsonL.mean(axis=1), pearsonL.std(axis=1), label='LO')
plt.errorbar(Ts, pearsonM.mean(axis=1), pearsonM.std(axis=1), label='MAP>0')
plt.xlabel('time samples')
plt.ylabel(r'$\rho(\mathbf{C}, \mathbf{C*})$')
plt.subplot(122)
plt.errorbar(Ts, pears_sigmaB.mean(axis=1), pears_sigmaB.std(axis=1), label='MAP')
plt.errorbar(Ts, pears_sigmaL.mean(axis=1), pears_sigmaL.std(axis=1), label='LO')
plt.xlabel('time samples')
plt.ylabel(r'$\rho(\mathbf{\Sigma}, \mathbf{\Sigma*})$')
plt.legend()

#%% estimation varying N
Ts = [300, 600, 3000]
Ms = [5, 10, 50]  # dimensionality
repetitions = 5
pearsonL = np.zeros([np.size(Ms), repetitions])
pearsonB = np.zeros([np.size(Ms), repetitions])
pearsonM = np.zeros([np.size(Ms), repetitions])
pears_sigmaL = np.zeros([np.size(Ms), repetitions])
pears_sigmaB = np.zeros([np.size(Ms), repetitions])

for i,m in enumerate(Ms):
    for r in range(repetitions):
        conn_strength = 8  #  connectivity strength varies with dimensionality
        X, C, S = MOU_sim(N=m, Sigma=None, T=Ts[i], connectivity_strength=conn_strength)
        Cl, Sl, taul = MOU_Lyapunov(X, verbose=0)
        Cb, Sb, taub = map_MOU(X.T, verbose=0)
        Cm = Cb.copy()
        Cm[Cm<0] = 0
        pearsonL[i, r] = pearsonr(Cl.flatten(), C.flatten())[0]
        pearsonB[i, r] = pearsonr(Cb.flatten(), C.flatten())[0]
        pearsonM[i, r] = pearsonr(Cm.flatten(), C.flatten())[0]
        pears_sigmaL[i, r] = pearsonr(Sl.diagonal(), S.diagonal())[0]
        pears_sigmaB[i, r] = pearsonr(Sb.diagonal(), S.diagonal())[0]

plt.figure()
plt.subplot(121)
plt.errorbar(Ms, pearsonB.mean(axis=1), pearsonB.std(axis=1), label='MAP')
plt.errorbar(Ms, pearsonL.mean(axis=1), pearsonL.std(axis=1), label='LO')
plt.errorbar(Ms, pearsonM.mean(axis=1), pearsonM.std(axis=1), label='MAP>0')
plt.xlabel('# nodes')
plt.ylabel(r'$\rho(\mathbf{C}, \mathbf{C*})$')
plt.subplot(122)
plt.errorbar(Ms, pears_sigmaB.mean(axis=1), pears_sigmaB.std(axis=1), label='MAP')
plt.errorbar(Ms, pears_sigmaL.mean(axis=1), pears_sigmaL.std(axis=1), label='LO')
plt.xlabel('# nodes')
plt.ylabel(r'$\rho(\mathbf{\Sigma}, \mathbf{\Sigma*})$')
plt.legend()
#%% estimation varying connectivity strenght
Ss = [1.2, 1.5, 1.8]
m = 50  # dimensionality
repetitions = 5
pearsonL = np.zeros([np.size(Ss), repetitions])
pearsonB = np.zeros([np.size(Ss), repetitions])
for i,s in enumerate(Ss):
    for r in range(repetitions):
        conn_strength = s/m*(m**2*.2-m)/2  # to get approx connectivity strength in the same range when varying dimensionality
        X, C, S = MOU_sim(N=m, Sigma=None, T=1600, connectivity_strength=conn_strength)
        Cl, Sl, taul = MOU_Lyapunov(X, verbose=0)
        Cb, Sb, taub = map_MOU(X.T, verbose=0)
        pearsonL[i, r] = pearsonr(Cl.flatten(), C.flatten())[0]
        pearsonB[i, r] = pearsonr(Cb.flatten(), C.flatten())[0]

plt.figure()
plt.errorbar(Ss, pearsonB.mean(axis=1), pearsonB.std(axis=1), label='MAP')
plt.errorbar(Ss, pearsonL.mean(axis=1), pearsonL.std(axis=1), label='LO')
plt.xlabel('connectivity scale')
plt.ylabel(r'$\rho(\mathbf{C}, \mathbf{C*})$')
plt.legend()