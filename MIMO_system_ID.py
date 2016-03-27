# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:56:03 2016

@author: User
"""

from __future__ import division
import numpy as np
from control import tf, ss
from matplotlib import pyplot as plot
from numpy import linalg



#-----------------------------------------

#G1
G1 = [[1],[5,1]]
#G2
G2 = [[1],[3,1]]
#g3
G3 = [[1],[2,1]]
#g4
G4 = [[1],[3,1]]

sys_G1 = ss(tf(G1[0],G1[1]))
sys_G2 = ss(tf(G2[0],G2[1]))
sys_G3 = ss(tf(G3[0],G3[1]))
sys_G4 = ss(tf(G4[0],G4[1]))


A_G1, B_G1, C_G1, D_G1 = np.asarray(sys_G1.A), np.asarray(sys_G1.B), np.asarray(sys_G1.C), \
    np.asarray(sys_G1.D)

A_G2, B_G2, C_G2, D_G2 = np.asarray(sys_G2.A), np.asarray(sys_G2.B), np.asarray(sys_G2.C), \
    np.asarray(sys_G2.D)
    
A_G3, B_G3, C_G3, D_G3 = np.asarray(sys_G3.A), np.asarray(sys_G3.B), np.asarray(sys_G3.C), \
    np.asarray(sys_G3.D)

A_G4, B_G4, C_G4, D_G4 = np.asarray(sys_G4.A), np.asarray(sys_G4.B), np.asarray(sys_G4.C), \
    np.asarray(sys_G4.D)

    
Nstates_G1 = A_G1.shape[0]
Nstates_G2 = A_G2.shape[0]
Nstates_G3 = A_G3.shape[0]
Nstates_G4 = A_G4.shape[0]

z_1 = np.zeros((Nstates_G1, 1))
z_2 = np.zeros((Nstates_G2, 1)) 
z_3 = np.zeros((Nstates_G3, 1)) 
z_4 = np.zeros((Nstates_G4, 1))   

tstart = 0
tend = 100
dt = 0.01

tspan = np.arange(tstart, tend, dt)


yplot = []
ybplot = []
sigma = 0.001

next_time = 0
j = 0

delta = 1.0



input1 = 1.0
input2 = 0

next_timeA = 0
next_timeB = delta
next_timeC = 4*delta

phi_T = []
y_list = []

phi2_T = []
y2_list = []

my_sum = np.zeros((6,6))
my_sum2 = np.zeros((6,1))

my_sumB = np.zeros((6,6))
my_sumB2 = np.zeros((6,1))


yb = yb_1 = yb_2 = 0
y = y_1 = y_2 = 0
input1_1 = input1_2 = 0
input2_1 = input2_2 = 0


for i ,t in enumerate(tspan):
    
    noise = sigma*np.random.rand()
    
    
    if t >= next_time:
        cnt = (-1)**j
        input1 += 2*cnt 
        input2 -= 2*cnt
        j += 1 
        delta2 = 2        
        next_time += delta2
         
    #-------------------------------------------------------
    if t >= next_timeA:
        if t >= next_timeB:
            phi_T.append([y_1, y_2, input1_1, input1_2, input2_1, input2_2])

            phi = np.matrix.transpose(np.array(phi_T))
            y_list.append([y])
            product = np.dot(phi, phi_T)
            product2 = np.dot(phi, y_list)
            
            my_sum += product
            my_sum2 += product2
            ###########################################
            phi2_T.append([yb_1, yb_2, input1_1, input1_2, input2_1, input2_2])

            phi2 = np.matrix.transpose(np.array(phi2_T))
            y2_list.append([yb])
            productB = np.dot(phi2, phi2_T)
            productB2 = np.dot(phi2, y2_list)
            
            my_sumB += productB
            my_sumB2 += productB2
            
            
            next_timeB += delta
    
        y_2 = y_1
        y_1 = y
        
        yb_2 = yb_1
        yb_1 = yb
        
        input1_2 = input1_1
        input2_2 = input2_1
        input1_1 = input1
        input2_1 = input2
        
        phi_T = []
        y_list = []
        
        phi2_T = []
        y2_list = []
        
        
        next_timeA += delta
    if t >= next_timeC:
        r = linalg.det(my_sum)
        s = linalg.inv(my_sum)
        parameters = np.dot(s, my_sum2)
        
        r2 = linalg.det(my_sumB)
        s2 = linalg.inv(my_sumB)
        parameters2 = np.dot(s2, my_sumB2)
        
        next_timeC += delta
    #-------------------------------------------------------
    
    dzdt1 = A_G1*z_1 + B_G1*input1
    y1 = C_G1*z_1 + D_G1*input1
    
    dzdt2 = A_G2*z_2 + B_G2*input2
    y2 = C_G2*z_2 + D_G2*input2
    
    y = y1 + y2
    y = y[0,0] + noise
    
    dzdt3 = A_G3*z_3 + B_G3*input1
    y3 = C_G3*z_3 + D_G3*input1
    
    dzdt4 = A_G4*z_4 + B_G4*input2
    y4 = C_G4*z_4 + D_G4*input2
    
    yb = y3 + y4 
    yb = yb[0,0] + noise
    
    z_1 += dzdt1*dt
    z_2 += dzdt2*dt
    z_3 += dzdt3*dt
    z_4 += dzdt4*dt
    
    yplot.append(y)
    ybplot.append(yb)
    
b1 = np.exp(-delta/5)
b2 = np.exp(-delta/3)
a1 = 1 - b1
a2 = 1 - b2


a_1 = b1 + b2
a_2 = b1*b2
a_3 = a1
a_4 = a1*b2
a_5 = a2
a_6 =  a2*b1

a, b, c, d, e, f = parameters[:,0]

bb1 = np.exp(-delta/2)
bb2 = np.exp(-delta/3)
ab1 = 1 - b1
ab2 = 1 - b2


ab_1 = bb1 + bb2
ab_2 = bb1*bb2
ab_3 = ab1
ab_4 = ab1*bb2
ab_5 = ab2
ab_6 =  ab2*bb1

ab, bb, cb, db, eb, fb = parameters2[:,0]

print ([a, b, c, d, e, f])
print ([a_1, -a_2, a_3, -a_4, a_5, -a_6])
print()
print()
print ([ab, bb, cb, db, eb, fb])
print ([ab_1, -ab_2, ab_3, -ab_4, ab_5, -ab_6])
    
