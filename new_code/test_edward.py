#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:02:06 2017

@author: andrea

Edward takes a lot of time for contructing the inference objects since it
builds the computational graph. This time seems to be in the order of 2 minutes
for 300 timesteps independently of the dimension of the system. However 
it takes really long time (more than 15 min but I killed the process) 
if the inference is constructed again in the same console (even after clearing
variables). So it is better to run the script in a dedicated console instead of lazily execute cells.
"""

from MOU_simulation import MOU_sim
from MOU_estimation import MOU_Lyapunov
from MOU_bayes import map_MOU 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import logm
from scipy.stats import pearsonr
import timeit
sns.set_context('notebook')
#%% try VI with Edward
# TODO: beta is actually the expm of C: correct this!

import edward as ed
import tensorflow as tf
from edward.models import Normal, InverseGamma, PointMass, Uniform, TransformedDistribution
# simulate data
d = 50
T = 300
X, C, S = MOU_sim(N=d, Sigma=None, mu=0, T=T, connectivity_strength=8.)

# the model
mu = tf.constant(0.)  # Normal(loc=tf.zeros([d]), scale=1.*tf.ones([d]))
beta = Normal(loc=tf.ones([d, d]), scale=2.*tf.ones([d,d]))
ds = tf.contrib.distributions
C = TransformedDistribution(
  distribution=beta,
  bijector=ds.bijectors.Exp(),
  name="LogNormalTransformedDistribution")

noise_proc = InverseGamma(concentration=tf.ones([d]), rate=tf.ones([d]))  # tf.constant(0.1)
noise_obs = tf.constant(0.1)  # InverseGamma(alpha=1.0, beta=1.0)

x = [0] * T
x[0] = Normal(loc=mu, scale=10.*tf.ones([d]))
for n in range(1, T):
    x[n] = Normal(loc=mu + tf.tensordot(C, x[n-1], axes=[[1],[0]]),
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
#qmu = Normal(loc=tf.Variable(tf.random_normal([d])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([d]))))
qbeta = Normal(loc=tf.Variable(tf.random_normal([d, d])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([d,d]))))
qnoise = InverseGamma(concentration=tf.Variable(tf.ones([d])), rate=tf.Variable(tf.ones([d])))
print("constructing inference object")

start_time = timeit.default_timer()
#inference_vb = ed.KLqp({beta: qbeta, mu: qmu}, data={xt: xt_true for xt, xt_true in zip(x, X)})
inference_vb = ed.KLqp({beta: qbeta, noise_proc: qnoise}, data={xt: xt_true for xt, xt_true in zip(x, X)})
elapsed = timeit.default_timer() - start_time
print("elapsed time is: ", elapsed)
print("running VB inference")
inference_vb.run()

Cvb = qbeta.mean().eval()

pp = qbeta.cdf(0.).eval()
Cvb_filt = Cvb.copy()
Cvb_filt[pp>0.05] = 0
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