# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:22:49 2016

@author: Charles
"""

from __future__ import division
import numpy as np
from control.matlab import tf
from matplotlib import pyplot as plot
from control.statesp import _convertToStateSpace
# Exact Process Model
K = 10
tau = 2
numer = [K]
denom = [tau,1]
sys1 = tf(numer,denom)

#Transformation to State-Space

sys = _convertToStateSpace(sys1)
A, B, C, D = np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C), \
    np.asarray(sys.D)

Nstates = A.shape[0]
z = np.zeros((Nstates, 1))

t_start = 0
t_end = 500
dt = 0.05
tspan = np.arange(t_start, t_end, dt)
npoints = len(tspan)

#List for storages
yplot = np.zeros(npoints)
ym_plot = np.zeros(npoints)


deviation = 1


#Initial conditions
z = 0
y = y_1 = 0

#Sampling parameters
next_time = 0
delta = dt + 1  # sampling interval

#new list of values to be  sampled
sampled_data = []

#sampling instances
t_sampling = []

#Parameters for PRBS simulation
next_time2 = 0

#initial signal
signal = -1

#store values of PRBS signal
sig_list = []

#initialization of the counter
j = 0

#storage of SAMPLED PRBS signal
input_signal_list = [] 
  
mode_signal = 0

#Initial Estimates
Q_0 = [[0.],
       [0.]]
constant = 100
P_0 = constant*np.array([[1,0],
                         [0,1]])
                         
phi_T = []
constant2 = 0.0
Q = []         
Q2 = []
R_1 = constant2*np.array([[200,-0.5],
                          [100,1]])   
lambd = 1.0

a = np.exp(-dt/tau)
b = K*(1-a)

a_list = []
b_list = []
yd = 0
yd_list = []

for i, t in enumerate(tspan):
    # simulation of the PRBS signal\
    
    signal2 = signal
    e_t = deviation*np.random.rand()
    if mode_signal == 0:
        if t >= next_time2:
            
            #Alternator (1 and -1)
            cnt = (-1)**j
            
            #add alternator to the initial signal 
            signal += 2*cnt
            
            #counter
            j += 1
            
            #random generator for width/span of the signal 
            delta2 = np.random.randint(2,10)
            
            next_time2 += delta2
    elif mode_signal == 1:
        signal = 1
        
    noise = deviation*np.random.rand()
     
    
    #Sampling the output signal and time instances
    if t >= next_time:
        a_list.append(a)
        b_list.append(b)
        ym = y_1
        phi_T.append([ym, signal])
        phi = np.matrix.transpose(np.array(phi_T))
        product_1 = np.dot(P_0, phi)
        product_2 = np.dot(np.dot(phi_T, P_0), phi)
        K_t = (1/(lambd + product_2))*product_1  
        #print (product_2)
        P_t = (P_0 - np.dot(np.dot(K_t,phi_T), P_0))*(1/lambd) + R_1 
        e_t = y - np.dot(phi_T, Q_0) 
        
        Q_t = Q_0 + e_t*K_t
        Q.append(Q_t[0,0])
        Q2.append(Q_t[1,0])
        Q_0 = Q_t
        P_0 = P_t
        
        sampled_data.append(y)
        input_signal_list.append(signal)
        t_sampling.append(t)
        phi_T = []
        next_time += delta
        
    y_1 = y 
    
    y1 = yd
    yd = Q_t[0][0]*y1 + Q_t[1][0]*signal2
    yd_list.append(yd)
    
    dzdt = A*z + B*signal    
    y = C*z + D*signal
    y += noise
    z += dzdt*dt
    yplot[i] = y[0,0] - noise
    ym_plot[i] = y
    sig_list.append(signal)
    

tau = -delta/np.log(Q_t[0,0])
print ('a = ',Q_t[0][0][0][0])
print ('b = ',Q_t[1][0][0][0])
K = Q_t[1][0]/(1 - Q_t[0,0])


print ('a = ',a)
print ('b = ', b)

plot.subplot(2,2,1)
plot.plot(t_sampling, Q, t_sampling, a_list )
plot.subplot(2,2,3)
plot.plot(t_sampling, Q2, t_sampling, b_list)
plot.subplot(2,2,2)
plot.plot(tspan, yplot, tspan, yd_list)
plot.subplot(2,2,4)
plot.plot(tspan, sig_list)

plot.show()
