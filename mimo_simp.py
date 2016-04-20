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
        

u_1, u ,v, v_1, y_1, z_1, y, z = 0,0,0,0,0,0,0,0
outputs = []
tstart = 0
tend = 100
T = 1.0
tspan = np.arange(tstart, tend, T)


b = np.exp(-T/5)
b_2 = np.exp(-T/2)
a_1 = 1 - b
a_2 = 0.5*a_1
a_4 = 1 - b_2
a_3 = 0.1*a_4

Q_real = []
Q_esti = []
Q2_esti = []

#controller
Kc = 1.0
tau_i = 1
tau_d = 1.1

K = 1.
tau2_i = 1
tau2_d = 0.6
e_2, e_1,  e, e2_2, e2_1, e2 = [0,0,0,0,0,0]
sigma = 0.05
#ID
sigma2 = 1000000000000

Q_0 = np.zeros((3,1))
P_0 = sigma2*np.eye(3)
Q2_0 = np.zeros((3,1))
P2_0 = sigma2*np.eye(3)
lambd = 1.0

phi_T = []
y_list = []

phi2_T = []
z_list = []
next_time = 0
j = 0

ysp = 0.5
ysp2 = 0.1
yk_1, yk, zk_1, zk = [0,0,0,0]
   
for t in tspan:
    
    noise = sigma*np.random.rand()
    noise2 = sigma*np.random.rand()
    if t >= next_time:
        cnt = (-1)**j
        ysp += 1*cnt 
        ysp2 += 1*cnt
        j += 1 
        delta2 = 20
        next_time += delta2
    
    outputs.append([y,z,yk,zk])
    Q_real.append([b,a_1,a_2,b_2,a_3,a_4])    
    Q_esti.append(Q_0.T[0])
    Q2_esti.append(Q2_0.T[0])
#----------------DIENTIFICATION---------------------------------------------

    phi_T.append([y_1, u_1, v_1])
    phi2_T.append([z_1, u_1, v_1])
    
    phi = np.matrix.transpose(np.array(phi_T))
    phi2 = np.matrix.transpose(np.array(phi_T))
    
    y_list.append([y])
    z_list.append([z])
    
    alpha = np.dot(np.dot(P_0,phi),np.dot(phi_T,P_0))
    beta = lambd + np.dot(np.dot(phi_T,P_0),phi)
    
    alpha2 = np.dot(np.dot(P2_0,phi2),np.dot(phi2_T,P2_0))
    beta2 = lambd + np.dot(np.dot(phi2_T,P2_0),phi2)
    
    P_t = (P_0 - alpha/beta)/lambd
    P2_t = (P2_0 - alpha2/beta2)/lambd
    
    K_t = np.dot(P_t,phi)
    K2_t = np.dot(P2_t,phi2)
    
    e_t = y - np.dot(phi_T,Q_0)
    e2_t = z - np.dot(phi2_T,Q2_0)
    
    Q_t = Q_0 + np.dot(K_t,e_t)
    Q2_t = Q2_0 + np.dot(K2_t,e2_t)
    
    Q_0 = Q_t
    P_0 = P_t
    
    Q2_0 = Q2_t
    P2_0 = P2_t
    
    phi_T = []
    y_list = []
    phi2_T = []
    z_list = []
 #---------------------------------------------------------------------------   
    e_2 = e_1
    e_1 = e
    e = ysp - y
    
    e2_2 = e2_1
    e2_1 = e2
    e2 = ysp2 - z
        
    u = u_1 + Kc*((e-e_1) + (T/tau_i)*e + (tau_d/T)*(e - 2*e_1 + e_2)) 
    v = v_1 + K*((e2-e2_1) + (T/tau2_i)*e2 + (tau2_d/T)*(e2 - 2*e2_1 + e2_2))
    
    y_1 = y
    z_1 = z
    u_1 = u
    v_1 = v
    yk_1 = yk
    zk_1 = zk    
    
    yk = Q_t[0,0]*yk_1 + Q_t[1,0]*u_1 + Q_t[2,0]*v_1    
    zk = Q2_t[0,0]*zk_1 + Q2_t[1,0]*u_1 + Q2_t[2,0]*v_1 
    
    y = b*y_1 + a_1*u_1 + a_2*v_1  + noise    
    z = b_2*z_1 + a_3*u_1 + a_4*v_1 + noise2
        
Q_esti = np.array(Q_esti) 
Q_real = np.array(Q_real)   
outputs = np.array(outputs)
 
plot.subplot(2,1,1)
plot.plot(tspan, outputs)
plot.ylabel('outputs')
plot.subplot(2,1,2)
plot.plot(tspan, Q_esti, tspan, Q2_esti)
plot.plot(tspan, Q_real, 'k')
plot.xlabel('time')
plot.ylabel('Parameters')
plot.show()