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
tend = 50
T = 1.0
tspan = np.arange(tstart, tend, T)


b = np.exp(-T/5)
b_2 = np.exp(-T/2)

a_1 = 1 - b
a_2 = 0.5*a_1
a_4 = 1 - b_2
a_3 = 0.1*a_4

for t in tspan:
    u = step(0, 0.1, 0, t)
    v = step(0, 0.1, 0, t)
    yplot.append(y)
    zplot.append(z)
    y = b*y_1 + a_1*u_1 + a_2*v_1    
    z = b*z_1 + a_3*u_1 + a_4*v_1
    
    y_1 = y
    z_1 = z
    u_1 = u
    v_1 = v
plot.plot(tspan, yplot)
plot.plot(tspan, zplot)
plot.show()