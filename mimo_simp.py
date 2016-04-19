# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:57:36 2016

@author: StudyCentre
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plot

def step(start, step, tstep, t):
    if t >= tstep:
        return start + step
    else:
        return start
        
u_1 = u = 0
v_1 = v = 0
y_1 = y = 0
z_1 = z = 0

yplot = []
zplot = []

tstart = 0
tend = 300
T = 1.0
tspan = np.arange(tstart, tend, T)


b = np.exp(-T/5)
b_2 = np.exp(-T/2)

a_1 = 1 - b
a_2 = 0.5*a_1
a_4 = 1 - b_2
a_3 = 0.1*a_4


#controller
Kc = 1.0
tau_i = 1
tau_d = 1.1

K = 1.
tau2_i = 1
tau2_d = 0.6

e_2 = e_1 = e = 0
e2_2 = e2_1 = e2 = 0

sigma = 0.005


#ID
Q_0 = np.zeros((3,1))
sigma2 = 1000000000000
P_0 = sigma2*np.eye(3)
lambd = 1.0
phi_T = []
y_list = []
next_time = 0
j = 0
#w_1 = w2_1 = w = 0
ysp = 0
ysp2 = 0.5
for t in tspan:
    
    noise = sigma*np.random.rand()
    noise2 = sigma*np.random.rand()
    #ysp = step(0, 0.2, 0, t)
    #ysp2 = step(0.15, -0.05, 10, t)
    if t >= next_time:
        cnt = (-1)**j
        ysp += 0.05*cnt 
        ysp2 += 0.05*cnt
        j += 1 
        delta2 = 20
        next_time += delta2
    
    yplot.append(y)
    zplot.append(z)
    
#----------------DIENTIFICATION---------------------------------------------
    phi_T.append([y_1, u_1, v_1])

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
    
    phi_T = []
    y_list = []

 #----------------------------------------------------------------   
    e_2 = e_1
    e_1 = e
    e = ysp - y
    
    e2_2 = e2_1
    e2_1 = e2
    e2 = ysp2 - z
    u = u_1 + Kc*((e-e_1) + (T/tau_i)*e + (tau_d/T)*(e - 2*e_1 + e_2)) + noise
    v = v_1 + K*((e2-e2_1) + (T/tau2_i)*e2 + (tau2_d/T)*(e2 - 2*e2_1 + e2_2)) + noise2
    
    y = b*y_1 + a_1*u_1 + a_2*v_1  
    
    z = b_2*z_1 + a_3*u_1 + a_4*v_1
    
    y_1 = y
    z_1 = z
    u_1 = u
    v_1 = v
    
print Q_0
print b, a_1, a_2   
plot.plot(tspan, yplot)
plot.plot(tspan, zplot)
plot.show()