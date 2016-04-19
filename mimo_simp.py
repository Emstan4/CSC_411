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
        
u_1 = 0
v_1 = 0
y_1 = y = 0
z_1 = z = 0
u = 0.8
v = 0.1
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


sigma = 0.008

Q_0 = np.zeros((3,1))
sigma2 = 100000000000000      
P_0 = sigma2*np.eye(3)
lambd = 1.0

phi_T = []
y_list = []
next_time = 0
next_time2 = T
j = 0

my_sum = np.zeros((3,3))
my_sum2 = np.zeros((3,1))
for t in tspan:
    noise = sigma*np.random.rand()
#    u = step(0, 0.1, 0, t)
#    v = step(0.5, -0.05, 20, t)
    if t >= next_time:
        cnt = (-1)**j
        u += 1.5*cnt 
        v -= 1*cnt
        j += 1 
        next_time += 10
        
    yplot.append(y)
    zplot.append(z)
    
    if t >= next_time2:
        phi_T.append([y_1, u_1, v_1])
        phi = np.matrix.transpose(np.array(phi_T))
        y_list.append([y])
        product = np.dot(phi, phi_T)
        product2 = np.dot(phi, y_list)
        my_sum += product
        my_sum2 += product2
        phi_T = []
        y_list = []
        next_time2 += T

    y_1 = y
    z_1 = z
    u_1 = u
    v_1 = v
    y = b*y_1 + a_1*u_1 + a_2*v_1 + noise    
    z = b_2*z_1 + a_3*u_1 + a_4*v_1 + noise
    
    
    


r = np.linalg.inv(my_sum)
parameters = np.dot(r, my_sum2)
     
print parameters 
print b, a_1, a_2  
plot.plot(tspan, yplot)
plot.plot(tspan, zplot)
plot.show()