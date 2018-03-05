import os, sys
import numpy as np


os.system('clear')


##################
# model parameters

T = 9000. # duration of simulation
T0 = 100. # initialization time for network dynamics

dt = 0.05 # temporal resolution for simulation
n_sampl = int(1./dt) # sampling to get 1 point every second
tau_x = 1. # time constant of dynamical system

# number of network nodes
N = 50

# input noise matrix (diagonal only so far)
coef_Sigma_diag = 1.
coef_Sigma_offdiag = 0  # 0.2


# activation function Phi
act_mode = 'lin'
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
	print('wrong act mode')
	sys.exit()


# artificial network with connectivity ollowing structure in SC matrix
C_orig = np.random.rand(N,N)
C_orig[np.random.rand(N,N)>0.2] = 0
C_orig[np.eye(N, dtype=bool)] = 0
C_orig *= 4. * N / C_orig.sum()


# input mean
I_orig = np.random.rand(N)

# input noise matrix from mixing Gaussian White noise
Sigma_mixing = coef_Sigma_offdiag * (2*np.random.rand(N,N)-1)
Sigma_mixing[np.random.rand(N,N)>0.1] = 0
Sigma_mixing[np.eye(N,dtype=bool)] = coef_Sigma_diag * (0.5+0.5*np.random.rand(N))
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

# save data
np.save('ts_emp.npy',ts_emp[::n_sampl,:])

np.save('C_orig.npy',C_orig)
np.save('Sigma_orig.npy',Sigma_orig)

