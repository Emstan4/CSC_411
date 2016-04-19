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
tend = 100
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

sigma = 0.01



for t in tspan:
    
    noise = sigma*np.random.rand()
    noise2 = sigma*np.random.rand()
    ysp = step(0, 0.2, 0, t)
    ysp2 = step(0.15, -0.05, 40, t)
    
    yplot.append(y)
    zplot.append(z)
    
    e_2 = e_1
    e_1 = e
    e = ysp - y
    
    e2_2 = e2_1
    e2_1 = e2
    e2 = ysp2 - z
    u = u_1 + Kc*((e-e_1) + (T/tau_i)*e + (tau_d/T)*(e - 2*e_1 + e_2)) + noise
    v = v_1 + K*((e2-e2_1) + (T/tau2_i)*e2 + (tau2_d/T)*(e2 - 2*e2_1 + e2_2)) + noise2
    
    y = b*y_1 + a_1*u_1 + a_2*v_1    
    z = b*z_1 + a_3*u_1 + a_4*v_1
    
    y_1 = y
    z_1 = z
    u_1 = u
    v_1 = v
    
plot.plot(tspan, yplot)
plot.plot(tspan, zplot)
plot.show()