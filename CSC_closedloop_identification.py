# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:58:42 2016

@author: Charles
"""

from __future__ import division
import numpy as np
from control.matlab import tf
from matplotlib import pyplot as plot
from numpy.linalg import lstsq
from control.statesp import _convertToStateSpace
# Exact Process Model
K = 10

numer = [1]
denom = [5,1]
sys1 = tf(numer,denom)

#Transformation to State-Space

sys = _convertToStateSpace(sys1)
A, B, C, D = np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C), \
    np.asarray(sys.D)

Nstates = A.shape[0]
z = np.zeros((Nstates, 1))

t_start = 0
t_end = 20
dt = 0.05
tspan = np.arange(t_start, t_end, dt)
npoints = len(tspan)

#List for storages
yplot = np.zeros(npoints)
ym_plot = np.zeros(npoints)

#sigma
deviation = 0.5
#Initial conditions
z = 0
y = 0

#Sampling parameters
next_time = 0
delta = 0.3 # sampling interval

#new list of values to be  sampled
sampled_data = []
#sampling instances
t_sampling = []

#Parameters for PRBS simulation
next_time2 = 0
#initial signal
signal = 0

#store values of PRBS signal
sig_list = []

#initialization of the counter
j = 0

#storage of SAMPLED PRBS signal
input_signal_list = [] 
Kc = 7

def step(start, stepsize, tstep, t):
    if t <= tstep:
        return start
    else:
        return start + stepsize 

mode_signal = 1
Ysp = 1
for i, t in enumerate(tspan):
    
    
        
    noise = deviation*np.random.rand()
    #PRBS (setpoint)
    if t >= next_time2:
            #Alternator (1 and -1)
            cnt = (-1)**j
            
            #add alternator to the initial signal 
            Ysp += 2*cnt
            
            #counter
            j += 1
            
            #random generator for width/span of the signal 
            delta2 = np.random.randint(1,4)
            
            next_time2 += delta2
    #Sampling the output signal and time instances
    if t > next_time:
        ym = y[0,0]
        sampled_data.append(y[0,0])
        input_signal_list.append(Ysp)
        t_sampling.append(t)
        next_time += delta
        
    
    dzdt = A*z + B*signal
    y = C*z + D*signal
    z += dzdt*dt
    
    # simulation of the PRBS signal\
    if mode_signal == 0:
        if t >= next_time2:
            #Alternator (1 and -1)
            cnt = (-1)**j
            
            #add alternator to the initial signal 
            signal += 2*cnt
            
            #counter
            j += 1
            
            #random generator for width/span of the signal 
            delta2 = np.random.randint(1,3)
            
            next_time2 += delta2
    elif mode_signal == 1:
        signal = Kc*(Ysp - y[0,0])
        
    yplot[i] = y[0,0]
    ym_plot[i] = ym + noise
    sig_list.append(Ysp)
plot.subplot(2,1,2)
plot.plot(tspan, ym_plot)
plot.subplot(2,1,1)
plot.plot(tspan, sig_list)
plot.show()

AA=np.zeros((len(sampled_data),2)) 
for i in range(len(sampled_data)):
    AA[i][0]=sampled_data[i-1]
    AA[i][1]=input_signal_list[i-1]
    AA[0][0]= 0
    AA[0][1]= 0
    
BB= np.zeros((len(sampled_data),1)) 
for i in range(len(sampled_data)):
    BB[i][0]= sampled_data[i]
    
    
X = AA
Y = BB
    
# transpose of matrix X
Xt =  np.matrix.transpose(np.array(X))        

#multiplication of matrix     
mat1 = np.dot(Xt, X)
mat2 = np.dot(Xt,Y)

#identity matrix
I = np.eye(len(mat1))

#solve for the inverse matrix of mat1 above
pre_beta = lstsq(mat1, I)[0]

#solving the list of process parameters according to the difference equation
beta = np.dot(pre_beta, mat2)
print (beta)
b = beta[1][0]/Kc
a = beta[0][0] + b*Kc
tau = -delta/np.log(a)
print (tau)
K = b/(1-a)
print (K)