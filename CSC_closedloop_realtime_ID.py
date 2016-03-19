# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:08:59 2016

@author: Charles
"""

from __future__ import division
import numpy as np
from control import tf, ss
from matplotlib import pyplot as plot

# Exact Process Model
K = 1
tau = 5
numer = [K]
denom = [tau,1]
sys1 = tf(numer,denom)

Kc = 2
tau_i = 5


#PI controller
controllerA = [Kc*tau_i, Kc]
controllerB = [tau_i, 0]

cont_sys1 = tf(controllerA, controllerB)
cont_sys = ss(cont_sys1)
#Transformation to State-Space

sys = ss(sys1)
A, B, C, D = np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C), \
    np.asarray(sys.D)

contA, contB, contC, contD = np.asarray(cont_sys.A), np.asarray(cont_sys.B), np.asarray(cont_sys.C), \
    np.asarray(cont_sys.D)
    
Nstates = A.shape[0]
contNstates = contA.shape[0]

z_cont = np.zeros((contNstates, 1))
z = np.zeros((Nstates, 1))

t_start = 0
t_end = 100
dt = 0.01
tspan = np.arange(t_start, t_end, dt)
npoints = len(tspan)

#List for storages
yplot = []
phi_T = []
#sigma
deviation = 0.0
#Initial conditions
z = 0
y = 0

#Sampling parameters

delta = 1# sampling interval
next_time = 0
next_time2 = delta

y_sp = 1

a = np.exp(-delta/tau)
b = K*(1 - a)
Q_real = [[a],
          [b]]
y_1 = 0
u = u_1 = 0
j = 0

my_sum = np.zeros((2,2))
my_sum2 = np.zeros((2,1))
y_list = []

deviation = 1.0
next_time3 = 0

#Real time initialization
Q_0 = np.zeros((2,1))
sigma = 1      
P_0 = sigma*np.eye(2)
lambd = 1.0


#List for storing data
Qplot_1 = []
Qplot_2 = []
t_plot = []
a_list = []
b_list = []

error_1 = np.zeros(npoints)
error_2 = np.zeros(npoints)

error = np.zeros((2,1))

int_listA = []
int_listB = []
cntA = 0
my_sumA = 1.0
my_sumB = 1.0
R = []
R2 = []
quality = 0
quality2 = 0
for i, t in enumerate(tspan):
    
    noise = deviation*np.random.rand()
    if t >= next_time3:
            cnt = (-1)**j
            y_sp += 2*cnt
            j += 1 
            delta2 = 5
            
            next_time3 += delta2
            
    if t >= next_time:
        t_plot.append(t)
        Qplot_1.append(Q_0[0,0])
        Qplot_2.append(Q_0[1,0])
        a_list.append(a)
        b_list.append(b)
        R.append(quality)
        R2.append(quality2)
        
        
                
        if quality <= quality2:            
            if quality >= 0.9:
                break
        elif quality >= quality2:            
            if quality2 >= 0.9:
                break
            
        cntA += 1
        if t >= next_time2:
                
            phi_T.append([y_1, u_1])
            
            phi = np.matrix.transpose(np.array(phi_T))
            y_list.append([y])
            
            alpha = np.dot(np.dot(P_0,phi),np.dot(phi_T,P_0))
            beta = lambd + np.dot(np.dot(phi_T,P_0),phi)
            
            P_t = (P_0 - alpha/beta)/lambd
            
            
            K_t = np.dot(P_t,phi)
            e_t = y - np.dot(phi_T,Q_0)
            Q_t = Q_0 + np.dot(K_t,e_t)
            
            Q_0 = Q_t
            P_0 = P_t
            
            #Quality determination
            V_Q = np.dot(np.matrix.transpose(y_list - np.dot(phi_T, Q_t)), y_list - np.dot(phi_T, Q_t))
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
            
            next_time2 += delta
        
        int_listA = []
        int_listB = []
        y_1 = y
        u_1 = u
        phi_T = []
        y_list = []
        
        
        next_time += delta
        
    yplot.append(y)
    e = y_sp - y
    
    dzcdt = contA*z_cont + contB*e
    u = contC*z_cont + contD*e
    u = u[0,0]
        
    dzdt = A*z + B*u
    y = C*z + D*u
    y = y[0,0]
    y += noise
    
    z += dzdt*dt
    z_cont += dzcdt*dt
    
plot.subplot(2,2,1)    
plot.plot(t_plot,R)
plot.subplot(2,2,3) 
plot.plot(t_plot, Qplot_2, t_plot, b_list)  

plot.subplot(2,2,2)    
plot.plot(t_plot,R2)
plot.subplot(2,2,4) 
plot.plot(t_plot, Qplot_1, t_plot, a_list) 
     
plot.show() 
