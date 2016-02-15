# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:23:49 2016

@author: Charles
"""

from __future__ import division
import numpy as np
from ChLab import state_space, writefile
from matplotlib import pyplot as plot

# Exact Process Model
numerator = [[0,1]]
denominator = [[5,1]]

#Transformation to State-Space
A, B, C, D = state_space(numerator, denominator)
Nstates = A.shape[0]
z = np.zeros((Nstates, 1))


t_start = 0
t_end = 30
dt = 0.1
tspan = np.arange(t_start, t_end, dt)
npoints = len(tspan)

#List for storages
yplot = np.zeros(npoints)
ym_plot = np.zeros(npoints)


deviation = 0.05


#Initial conditions
z = 0
y = 0

#Sampling parameters
next_time = 0
delta = 1 # sampling interval

#new list of values to be  sampled
sampled_data = []
#sampling instances
t_sampling = []

#Parameters for PRBS simulation
next_time2 = 0
#initial signal
signal = 1

#store values of PRBS signal
sig_list = []
#initialization of the counter
j = 0

#storage of SAMPLED PRBS signal
input_signal_list = [] 
   
for i, t in enumerate(tspan):
    # simulation of the PRBS signal
    if t >= next_time2:
        #Alternator (1 and -1)
        cnt = (-1)**j
        
        #add alternator to the initial signal 
        signal += cnt
        
        #counter
        j += 1
        
        #random generator for width/span of the signal 
        delta2 = np.random.randint(2,6)
        
        next_time2 += delta2
    
    noise = deviation*np.random.rand()
    yplot[i] = y 
    
    #Sampling the output signal and time instances
    if t > next_time:
        ym = y
        sampled_data.append(y[0,0])
        input_signal_list.append(signal)
        t_sampling.append(t)
        next_time += delta
        
    
    dzdt = A*z + B*(1)
    y = C*z + D*(1)
    z += dzdt*dt
    
    ym_plot[i] = ym + noise
    sig_list.append(signal)
plot.plot(tspan, ym_plot)
plot.show()

#arrange the time instances and the sampled output into one list 
AA=np.zeros((len(sampled_data),2)) 
for i in range(len(sampled_data)):
    AA[i][0]=sampled_data[i]
    AA[i][1]=input_signal_list[i]
    
#store the sampled data into a csv file to be retrieved by excel
writefile(AA,'CSC_Sampled_Data.csv')
print AA
print sampled_data
print input_signal_list
#prediction of process parameters is done on excel using the sampled data
