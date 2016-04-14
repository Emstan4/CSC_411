# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:56:58 2016

@author: StudyCentre
"""

from __future__ import division
import numpy as np
from control import tf, ss
from matplotlib import pyplot as plot


tau_1 = 5
tau_2 = 2
tau_3 = 3
tau_4 = 4

K_1 = 1
K_2 = 1
K_3 = 1
K_4 = 1

#controller 1
Kc = 3
tau_i = 3

#controller 2
Kc_2 = 1.0
tau_i_2 = 6

#PI controller
controllerA = [Kc*tau_i, Kc]
controllerB = [tau_i, 0]

cont_sys1 = tf(controllerA, controllerB)
cont_sys = ss(cont_sys1)

controllerA_2 = [Kc_2*tau_i_2, Kc_2]
controllerB_2 = [tau_i_2, 0]

cont_sys1_2 = tf(controllerA_2, controllerB_2)
cont_sys_2 = ss(cont_sys1_2)

#G1
G1 = [[K_1],[tau_1,1]]
#G2
G2 = [[K_2],[tau_2,1]]
#G3
G3 = [[K_3],[tau_3,1]]
#G4
G4 = [[K_4],[tau_4,1]]

sys_G1 = ss(tf(G1[0],G1[1]))
sys_G2 = ss(tf(G2[0],G2[1]))
sys_G3 = ss(tf(G3[0],G3[1]))
sys_G4 = ss(tf(G4[0],G4[1]))

contA, contB, contC, contD = np.asarray(cont_sys.A), np.asarray(cont_sys.B), np.asarray(cont_sys.C), \
    np.asarray(cont_sys.D)
    
contA_2, contB_2, contC_2, contD_2 = np.asarray(cont_sys_2.A), np.asarray(cont_sys_2.B), np.asarray(cont_sys_2.C), \
    np.asarray(cont_sys_2.D)

A_G1, B_G1, C_G1, D_G1 = np.asarray(sys_G1.A), np.asarray(sys_G1.B), np.asarray(sys_G1.C), \
    np.asarray(sys_G1.D)

A_G2, B_G2, C_G2, D_G2 = np.asarray(sys_G2.A), np.asarray(sys_G2.B), np.asarray(sys_G2.C), \
    np.asarray(sys_G2.D)
    
A_G3, B_G3, C_G3, D_G3 = np.asarray(sys_G3.A), np.asarray(sys_G3.B), np.asarray(sys_G3.C), \
    np.asarray(sys_G3.D)

A_G4, B_G4, C_G4, D_G4 = np.asarray(sys_G4.A), np.asarray(sys_G4.B), np.asarray(sys_G4.C), \
    np.asarray(sys_G4.D)

contNstates = contA.shape[0]    
contNstates_2 = contA_2.shape[0]
Nstates_G1 = A_G1.shape[0]
Nstates_G2 = A_G2.shape[0]
Nstates_G3 = A_G3.shape[0]
Nstates_G4 = A_G4.shape[0]

z_cont = np.zeros((contNstates, 1))
z_cont_2 = np.zeros((contNstates_2, 1))
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
sigma = 0.0

next_time = 0
j = 0

delta = 1.0



#input1 = 1.0
#input2 = 1.0

next_timeA = 0
next_timeB = delta


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


# Real time ID parameters

Q_0 = np.zeros((6,1))
sigma2 = 10000000000      
P_0 = sigma2*np.eye(6)
lambd = 1.0

qlist_a = []
qlist_b = []
qlist_c = []
qlist_d = []
qlist_e = []
qlist_f = []

tlist= []


     

b1 = np.exp(-delta/tau_1)
b2 = np.exp(-delta/tau_2)
a1 = K_1*(1 - b1)
a2 = K_2*(1 - b2)


a_1 = b1 + b2
a_2 = b1*b2
a_3 = a1
a_4 = a1*b2
a_5 = a2
a_6 = a2*b1

alist = []
blist = []  
clist = []
dlist = []
elist = []
flist = []   

Q_t = np.zeros((6,1))

y1_sp = 1.0
y2_sp = 1.0

for i ,t in enumerate(tspan):            
    noise = sigma*np.random.rand()     
    
    
#    if t >= next_time:
#        cnt = (-1)**j
#        input1 += 2*cnt 
#        input2 += 2*cnt
#        j += 1 
#        delta2 = 10
#        next_time += delta2
    
#    if t >= next_timeA:
#        qlist_a.append(Q_t[0,0])
#        qlist_b.append(Q_t[1,0])
#        qlist_c.append(Q_t[2,0])
#        qlist_d.append(Q_t[3,0])
#        qlist_e.append(Q_t[4,0])
#        qlist_f.append(Q_t[5,0])
#        
#        alist.append(a_1)
#        blist.append(-a_2)
#        clist.append(a_3)
#        dlist.append(-a_4)
#        elist.append(a_5)
#        flist.append(-a_6)
#        tlist.append(t)
#        if t >= next_timeB:
#            phi_T.append([y_1, y_2, input1_1, input1_2, input2_1, input2_2])
#
#            phi = np.matrix.transpose(np.array(phi_T))
#            y_list.append([y])
#            product = np.dot(phi, phi_T)
#            product2 = np.dot(phi, y_list)
#            
#            my_sum += product
#            my_sum2 += product2
#            
#            phi2_T.append([yb_1, yb_2, input1_1, input1_2, input2_1, input2_2])
#
#            phi2 = np.matrix.transpose(np.array(phi2_T))
#            y2_list.append([yb])
#
#            
#            alpha = np.dot(np.dot(P_0,phi),np.dot(phi_T,P_0))
#            beta = lambd + np.dot(np.dot(phi_T,P_0),phi)
#            
#            P_t = (P_0 - alpha/beta)/lambd
#            
#            
#            K_t = np.dot(P_t,phi)
#            e_t = y - np.dot(phi_T,Q_0)
#            Q_t = Q_0 + np.dot(K_t,e_t)
#            Q_0 = Q_t
#            P_0 = P_t
#            
#            productB = np.dot(phi2, phi2_T)
#            productB2 = np.dot(phi2, y2_list)
#            
#            my_sumB += productB
#            my_sumB2 += productB2
#            
#            
#            next_timeB += delta
#    
#        y_2 = y_1
#        y_1 = y
#        
#        yb_2 = yb_1
#        yb_1 = yb
#        
#        input1_2 = input1_1
#        input2_2 = input2_1
#        input1_1 = input1
#        input2_1 = input2
#        
#        phi_T = []
#        y_list = []
#        
#        phi2_T = []
#        y2_list = []
#        
#        
#        next_timeA += delta
    
    e = y1_sp - y

    e2 = y2_sp - yb
    
    dzcdt = contA*z_cont + contB*e
    u = contC*z_cont + contD*e
    u = u[0,0]
    input1 = u

    dzcdt_2 = contA_2*z_cont_2 + contB_2*e2
    u2 = contC_2*z_cont_2 + contD_2*e2
    u2 = u2[0,0]
    input1 = u
    input2 = u2
    
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
    z_cont += dzcdt*dt
    z_cont_2 += dzcdt_2*dt
    
    yplot.append(y)
    ybplot.append(yb)
    

#THESE PARAMETERS ARE FOR THE FIRST OUTPUT

#plot.subplot(6,1,1)
#plot.plot(tlist, qlist_a, tlist, alist)
#plot.ylabel("$c_1$", fontsize = 20)
#
#
#plot.subplot(6,1,2)
#plot.plot(tlist, qlist_b, tlist, blist)
#plot.ylabel("$c_2$", fontsize = 20)
#
#plot.subplot(6,1,3)
#plot.plot(tlist, qlist_c, tlist, clist)
#plot.ylabel("$c_3$", fontsize = 20)
#
#plot.subplot(6,1,4)
#plot.plot(tlist, qlist_d, tlist, dlist)
#plot.ylabel("$c_4$", fontsize = 20)
#
#plot.subplot(6,1,5)
#plot.plot(tlist, qlist_e, tlist, elist)
#plot.ylabel("$c_5$", fontsize = 20)
#
#plot.subplot(6,1,6)
#plot.plot(tlist, qlist_f, tlist, flist)
#plot.ylabel("$c_6$", fontsize = 20)
#plot.xlabel("time", fontsize = 20)
plot.plot(tspan, yplot,tspan, ybplot)
plot.show()