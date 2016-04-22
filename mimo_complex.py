# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:39:05 2016

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
        
#process parameters
tau_1, K_1 = 1,1
tau_2, K_2 = 5,0.5
tau_3, K_3 = 6,0.2
tau_4, K_4 = 2,1

T = 1.0
tstart = 0
tend = 100
tspan = np.arange(tstart, tend, T)

b_1 = np.exp(-T/tau_1)
b_2 = np.exp(-T/tau_2)
b_3 = np.exp(-T/tau_3)
b_4 = np.exp(-T/tau_4)

a_1 = K_1*(1 - b_1)
a_2 = K_2*(1 - b_2)
a_3 = K_3*(1 - b_3)
a_4 = K_4*(1 - b_4)

a = b_1 + b_3
b = -b_1*b_3
c = a_1
d = -a_1*b_3
e = a_3
f = -a_3*b_1

aa = b_4 + b_2
bb = -b_4*b_2
cc = a_4
dd = -a_4*b_2
ee = a_2
ff = -a_2*b_4

#lists        
outputs = []
inputs = []
para_estim = []
para_real = []

y, y_1, y_2 = 0,0,0
z, z_1, z_2 = 0,0,0
u, u_1, u_2 = 0,0,0
v, v_1, v_2 = 0,0,0
yk, yk_1, yk_2 = 0,0,0
zk, zk_1, zk_2 = 0,0,0

#controller
Kc = 0.1
tau_i = 5
tau_d = 0.0

Kcb = 0.5
taub_i = 2
taub_d = 0.0
e, e_1, e_2 = 0,0,0
eb, eb_1, eb_2 = 0,0,0

sigma = 0.02

#identification
#model: y(t) = a*y(t-1) + b*y(t-2) + c*u(t-1) + d*u(t-2) + e*v(t-1) + f*v(t-2)
#       z(t) = g*z(t-1) + h*z(t-2) + i*u(t-1) + j*u(t-2) + k*v(t-1) + l*v(t-2)

sigma2 = 1000000000
Q_0 = np.zeros((6,1))
P_0 = sigma2*np.eye(6)
Q2_0 = np.zeros((6,1))
P2_0 = sigma2*np.eye(6)
lambd = 1.0

phi_T = []
y_list = []

phi2_T = []
z_list = []
next_time = 0
j = 0
ysp = 2
ysp2 = 0
for t in tspan:
    
    noise = sigma*np.random.rand()
    noise2 = sigma*np.random.rand()
    
    outputs.append([y, z, yk, zk])
    para_estim.append(Q_0.T[0])
    para_real.append([a,b,c,d,e,f])
#    ysp = step(0.7,0,0,t)
#    ysp2 = step(1.,0,0,t)
    if t >= next_time:
        cnt = (-1)**j
        ysp += 0.3*cnt 
        ysp2 += 1*cnt
        j += 1 
        delta2 = 40
        next_time += delta2
    
    
    #Identification-------------------------------------------
    phi_T.append([y_1, y_2, u_1, u_2, v_1, v_2])
    phi2_T.append([z_1, z_2, u_1, u_2, v_1, v_2])
    
    phi = np.matrix.transpose(np.array(phi_T))
    phi2 = np.matrix.transpose(np.array(phi2_T))
    
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
    ########################################################
    e_2 = e_1
    e_1 = e
    er = ysp - y
    
    eb_2 = eb_1
    eb_1 = eb
    eb = ysp2 - z
    
    u = u_1 + Kc*((er-e_1) + (T/tau_i)*er + (tau_d/T)*(er - 2*e_1 + e_2)) 
    v = v_1 + Kcb*((eb-eb_1) + (T/taub_i)*eb + (taub_d/T)*(eb - 2*eb_1 + eb_2)) 
    
    y_2 = y_1
    y_1 = y
    z_2 = z_1
    z_1 = z
    u_2 = u_1
    u_1 = u
    v_2 = v_1
    v_1 = v
        
    y = a*y_1 + b*y_2 + c*u_1 + d*u_2 + e*v_1 + f*v_2 + noise
    z = aa*z_1 + bb*z_2 + cc*u_1 + dd*u_2 + ee*v_1 + ff*v_2 + noise2
    
    yk = Q_t[0,0]*yk_1 + Q_t[1,0]*yk_2 + Q_t[2,0]*u_1 + Q_t[3,0]*u_2 + Q_t[4,0]*v_1 + Q_t[5,0]*v_2
    zk = Q2_t[0,0]*zk_1 + Q2_t[1,0]*zk_2 + Q2_t[2,0]*u_1 + Q2_t[3,0]*u_2 + Q2_t[4,0]*v_1 + Q2_t[5,0]*v_2  
    yk_2 = yk_1
    yk_1 = yk
    zk_2 = zk_1
    zk_1 = zk
outputs = np.array(outputs)
para_real = np.array(para_real)
para_estim = np.array(para_estim)
plot.subplot(2,1,1)
plot.plot(tspan, outputs)
plot.subplot(2,1,2)
plot.plot(tspan, para_estim, tspan, para_real)
plot.show()