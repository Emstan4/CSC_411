# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:22:49 2016

@author: Charles
"""

from __future__ import division
import numpy as np
from control import tf, ss
from matplotlib import pyplot as plot

# Exact Process Model
K = 2
tau = 5
numer = [K]
denom = [tau,1]
sys1 = tf(numer,denom)

#Transformation to State-Space

sys = ss(sys1)
A, B, C, D = np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C), \
    np.asarray(sys.D)

Nstates = A.shape[0]
z = np.zeros((Nstates, 1))

t_start = 0
t_end = 300
dt = 0.01
tspan = np.arange(t_start, t_end, dt)
npoints = len(tspan)

#List for storages
yplot = []
deviation = 0.4


#Initial conditions
z = 0
y = y_1 = 0

#Sampling parameters
next_time = 0
delta = 1  # sampling interval

#sampling instances
t_sampling = []

#Parameters for PRBS simulation
next_time2 = 0

#initial signal
signal = -1
signal_1 = 0
#store values of PRBS signal
sig_list = []

#initialization of the counter
j = 0

mode_signal = 0

#Initial Estimates
Q_0 = np.zeros((2,1))
constant = 1000000
P_0 = constant*np.eye(2)
                         
phi_T = []

Q = []         
Q2 = []
lambd = 1.0

a = np.exp(-deltat/tau)
b = K*(1-a)
Q_real = [[a],[b]]
a_list = []
b_list = []
signal_1 = 0

error_1 = np.zeros(npoints)
error_2 = np.zeros(npoints)
cntA = 0
my_sumA = 1.0
my_sumB = 1.0
R = []
R2 = []
quality = 0
quality2 = 0
int_listA = []
int_listB = []
tplot = []

for i, t in enumerate(tspan):
    # simulation of the PRBS signal\
    sig_list.append(signal)
    tplot.append(t)
    yplot.append(y)
    
    if mode_signal == 0:
        if t >= next_time2:
            
            #Alternator (1 and -1)
            cnt = (-1)**j
            
            #add alternator to the initial signal 
            signal += 2*cnt
            
            #counter
            j += 1
            
            #random generator for width/span of the signal 
            delta2 = 10
            
            next_time2 += delta2
    elif mode_signal == 1:
        signal = 1
        
    noise = deviation*np.random.rand()
     
    
    #Sampling the output signal and time instances
    if t >= next_time:
        a_list.append(a)
        b_list.append(b)
        Q.append(Q_0[0,0])
        Q2.append(Q_0[1,0])
        R.append(quality)
        R2.append(quality2)
        t_sampling.append(t)
        
        phi_T.append([y_1, signal_1])
        phi = np.matrix.transpose(np.array(phi_T))
        product_1 = np.dot(P_0, phi)
        product_2 = np.dot(np.dot(phi_T, P_0), phi)
        K_t = (1/(lambd + product_2))*product_1  
        P_t = (P_0 - np.dot(np.dot(K_t,phi_T), P_0))*(1/lambd) 
        e_t = y - np.dot(phi_T, Q_0) 
        
        Q_t = Q_0 + e_t*K_t
        
        Q_0 = Q_t
        P_0 = P_t
        cntA +=1
        
        error = abs(Q_real - Q_t)
            
        error_1[cntA] = error[1,0]
        error_2[cntA] = error[0,0]
        
        for k in range(5):
            int_listA.append(error_1[cntA - k])  
            int_listB.append(error_2[cntA - k])
            
        my_sumA = np.sum(int_listA)
        my_sumB = np.sum(int_listB)
        
        quality = (1 - my_sumA)**2
        quality2 = (1 - my_sumB)**2
        
        phi_T = []
        int_listA = []
        int_listB = []
        y_1 = y
        signal_1 = signal
        next_time += delta

    dzdt = A*z + B*signal    
    y = C*z + D*signal
    y += noise
    z += dzdt*dt
    y = y[0,0]
    y = y + noise
    yplot[i] = y


plot.subplot(2,2,1)
plot.plot(t_sampling, Q, t_sampling, a_list )
plot.xlabel("time")
plot.ylabel("a", fontsize = 20)

plot.subplot(2,2,3)
plot.plot(t_sampling, Q2, t_sampling, b_list)
plot.xlabel("time")
plot.ylabel("b", fontsize = 20)

plot.subplot(2,2,2)
plot.plot(tspan, yplot)
plot.xlabel("time")
plot.ylabel("Y(t)", fontsize = 20)

plot.subplot(2,2,4)
plot.plot(tspan, sig_list)
plot.xlabel("time")
plot.ylabel("Input", fontsize = 20)

plot.show()
