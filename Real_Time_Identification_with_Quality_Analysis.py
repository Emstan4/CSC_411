# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 01:13:36 2016

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
t_end = 1000
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
delta = 1.1  # sampling interval

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
# most recent data points
Q_2 = Q_1 = Q_3 =Q_4 = Q_5 = Q_6=Q_7 = Q_8 = Q_9= 0

Q_k2 = Q_k1 = Q_k3 =Q_k4 = Q_k5 = Q_k6=Q_k7 = Q_k8 = Q_k9= 0

Q_t = np.array([[0],[0]])
 
#quality
error = []
Quality = []

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
        
        ym = y_1
        phi_T.append([ym, signal])
        phi = np.matrix.transpose(np.array(phi_T))
        product_1 = np.dot(P_0, phi)
        product_2 = np.dot(np.dot(phi_T, P_0), phi)
        K_t = (1/(lambd + product_2))*product_1  
        
        P_t = (P_0 - np.dot(np.dot(K_t,phi_T), P_0))*(1/lambd) + R_1 
        e_t = y - np.dot(phi_T, Q_0) 
        
        Q_10 = Q_9 #Q(t-10)
        Q_9 = Q_8 #Q(t-9)
        Q_8 = Q_7 #Q(t-8)
        Q_7 = Q_6 #Q(t-7)
        Q_6 = Q_5 #Q(t-6)
        Q_5 = Q_4 #Q(t-5)
        Q_4 = Q_3 #Q(t-4)
        Q_3 = Q_2 #Q(t-3)
        Q_2 = Q_1 #Q(t-2)
        Q_1 = Q_t[1,0] #Q(t-1)
        
        Q_k10 = Q_k9 #Q(t-10)
        Q_k9 = Q_k8 #Q(t-9)
        Q_k8 = Q_k7 #Q(t-8)
        Q_k7 = Q_k6 #Q(t-7)
        Q_k6 = Q_k5 #Q(t-6)
        Q_k5 = Q_k4 #Q(t-5)
        Q_k4 = Q_k3 #Q(t-4)
        Q_k3 = Q_k2 #Q(t-3)
        Q_k2 = Q_k1 #Q(t-2)
        Q_k1 = Q_t[0,0] #Q(t-1)
        
        Q_t = Q_0 + e_t*K_t
        #Sum of the errors of the most recent parameter estimates (10 recent)
        X = (abs(b - Q_10) +abs(b - Q_9) +abs(b - Q_8) +abs(b - Q_7) +abs(b - Q_6) +abs(b - Q_5) +abs(b - Q_4) +abs(b - Q_3) +abs(b - Q_2) + abs(b - Q_1) + abs(b - Q_t[1,0]))
        Y = (abs(a - Q_k10) +abs(a - Q_k9) +abs(a - Q_k8) +abs(a - Q_k7) +abs(a - Q_k6) +abs(a - Q_k5) +abs(a - Q_k4) +abs(a - Q_k3) +abs(a - Q_k2) + abs(a - Q_k1) + abs(a - Q_t[0,0]))
        
        R_squ = 1 - X
        R = 1 - Y
        
        Q.append(Q_t[0,0])
        Q2.append(Q_t[1,0])
        t_sampling.append(t)
        a_list.append(a)
        b_list.append(b)
        
        #R should range between 0 and 1
        if R_squ < 0:
            R_squ = 0
        if R < 0:
            R = 0
            
        error.append((R_squ))
        Quality.append((R))
        
        # Stop when the quality reaches 90 %
        if R_squ < R:
            if R_squ >= 0.90:
                break
        elif R_squ > R:
            if R >= 0.90:
                break
            
        Q_0 = Q_t
        P_0 = P_t
        
        sampled_data.append(y)
        input_signal_list.append(signal)
        
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
    


plot.subplot(2,2,1)
plot.plot(t_sampling, Q2, t_sampling, b_list)
plot.subplot(2,2,2)
plot.plot(t_sampling, Q, t_sampling, a_list)

plot.subplot(2,2,3)
plot.plot(t_sampling,error)
plot.ylim([0,1])
plot.subplot(2,2,4)
plot.plot(t_sampling,Quality)
plot.ylim([0,1])

plot.show()