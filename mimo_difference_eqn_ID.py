# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:39:05 2016

@author: charles
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plot
import scipy
from scipy import signal
import csv

tlist = []
ilist = []
#for o in range(6,50):
def step(start, step, tstep, t):
    if t >= tstep:
        return start + step
    else:
        return start
def square_wave(width, t):
    period = (np.pi)/width
    return scipy.signal.square(period*t, duty = 0.5)
    
def writefile(data, filename):        
    with open(filename, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        for row_data in data:
            csvwriter.writerow(row_data)
        
def readfile(filename):        
    data2 = []
    with open(filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for cnt, row in enumerate(csvreader):
            data2.append(float(row[0]))
    return data2 
    
para_offlineA =  readfile('off_para1.csv') 
para_offlineB =  readfile('off_para2.csv') 
para_offlineA = np.array([para_offlineA])
para_offlineB = np.array([para_offlineB])
  
#process parameters
tau_1, K_1 = 1,1
tau_2, K_2 = 5,0.5
tau_3, K_3 = 6,0.2
tau_4, K_4 = 2,1

T = 1.0
tstart = 0
tend = 100
tspan = np.arange(tstart, tend, T)
npoints = len(tspan)

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
eq = a_3
f = -a_3*b_1

aa = b_4 + b_2
bb = -b_4*b_2
cc = a_4
dd = -a_4*b_2
ee = a_2
ff = -a_2*b_4

#lists        
outputs = np.zeros((npoints, 6))
inputs = []
controller_outputs = np.zeros((npoints, 4))
para_estim = np.zeros((npoints,6))
para_estim2 = np.zeros((npoints,6))
para_real = np.zeros((npoints,6))
para_real2 = np.zeros((npoints,6))

yn, yn_1, yn_2 = 0,0,0
y, y_1, y_2 = 0,0,0
z, z_1, z_2 = 0,0,0
u, u_1, u_2 = 0,0,0
v, v_1, v_2 = 0,0,0
alp, alp_1, alp_2 = 0,0,0
bet, bet_1, bet_2 = 0,0,0
yk, yk_1, yk_2 = 0,0,0
zk, zk_1, zk_2 = 0,0,0
m, m_1, m_2 = 0,0,0
n, n_1, n_2 = 0,0,0
#controller
Kc = 0.1
tau_i = 5
tau_d = 0

Kcb = 0.5
taub_i = 2
taub_d = 0.0
e, e_1, e_2 = 0,0,0
eb, eb_1, eb_2 = 0,0,0

sigma = 0.01

#identification
#model: y(t) = a*y(t-1) + b*y(t-2) + c*u(t-1) + d*u(t-2) + e*v(t-1) + f*v(t-2)
#       z(t) = g*z(t-1) + h*z(t-2) + i*u(t-1) + j*u(t-2) + k*v(t-1) + l*v(t-2)
sigma2 = 10000000000
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
next_time2 = 0
j = 0
ysp = 0.6
ysp2 = 0.8

error_list = [1]
quality_list = []
error_sum = 1.0

np.random.seed(seed=0)
dist = 0
dist2 = 0

my_sumA_1 = np.zeros((6,6))
my_sumA_2 = np.zeros((6,1))
my_sumB_1 = np.zeros((6,6))
my_sumB_2 = np.zeros((6,1))


error = np.zeros(len(tspan))
error2 = np.zeros(len(tspan)) 
error3 = np.zeros(len(tspan))
error4 = np.zeros(len(tspan)) 
recent_error_list = []
counter = 0
quality = []
residues = []

nxt = T
j = 0

amp_2 = 0.02
amp_1 = 0.08
mode = 'smart'
for i, t in enumerate(tspan):
    
    noise = sigma*np.random.rand()
    noise2 = sigma*np.random.rand()
    
    outputs[i] = [y, z, yk, zk, m, n]
    inputs.append([alp, bet])
    para_estim[i] = Q_0.T[0]
    para_estim2[i] = Q2_0.T[0]
    para_real[i] = [a,b,c,d,eq,f]
    para_real2[i] = [aa,bb,cc,dd,ee,ff]            
    ysp = step(1.0,-0.0,50,t)
    ysp2 = step(0.5,0.0,70,t) 
    
    #PRBS of the disturbance
    s = np.random.randint(10,20)
    s2 = np.random.randint(10,20)
    #print s, s2
    
    
    if mode == 'smart':
        if t >= 0 and t <= 15:
            dist  = 0*square_wave(s,t)
            dist2 = amp_2**square_wave(s2,t)
        elif t >15 and t< 30:
            dist  = amp_1*square_wave(s,t)
            dist2 = 0*square_wave(s2,t)
        else:
            dist  = 0*square_wave(s,t)
            dist2 = 0*square_wave(s2,t)
    elif mode == 'normal':
        if t >= 0 and t <= 30:
            dist  = amp_1*square_wave(s,t)
            dist2 = amp_2*square_wave(s2,t)
#    if t >= nxt:
#        cnt = (-1)**j
#        dist += 2*cnt 
#        dist2 -= 2*cnt
#        j += 1 
#        delta = np.random.randint(1,3)        
#        nxt += delta
        
    #Identification-------------------------------------------
    
    phi_T.append([y_1, y_2, alp_1, alp_2, bet_1, bet_2])
    phi2_T.append([z_1, z_2, alp_1, alp_2, bet_1, bet_2])
    
    phi = np.matrix.transpose(np.array(phi_T))
    phi2 = np.matrix.transpose(np.array(phi2_T))
    
    y_list.append([y])
    z_list.append([z])
    
    productA_1 = np.dot(phi, phi_T)
    productA_2 = np.dot(phi, y_list)
    my_sumA_1 += productA_1
    my_sumA_2 += productA_2    
    
    
    productB_1 = np.dot(phi2, phi2_T)
    productB_2 = np.dot(phi2, z_list)
    my_sumB_1 += productB_1
    my_sumB_2 += productB_2
    
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
    
    
    ########################################################
    if t >= next_time2:        

        q_sampled = Q_t
        q_sampled2 = Q2_t
        
        next_time2 += 15
#    lst1 =  abs(np.array(outputs)[:,0] - np.array(outputs)[:,2])[i] 
#    lst2 = (np.array(outputs)[:,0])[i]
#    print lst1/lst2
    
    error[i] = abs(np.array(outputs)[:,0] - np.array(outputs)[:,2])[i]/(np.array(outputs)[:,0])[i] # error in the y-outputs(first)
    error2[i] = abs(np.array(outputs)[:,0] - np.array(outputs)[:,4])[i]/(np.array(outputs)[:,0])[i] # error in thez-outputs(second)
    error3[i] = abs(np.array(outputs)[:,1] - np.array(outputs)[:,3])[i]/(np.array(outputs)[:,1])[i]
    error4[i] = abs(np.array(outputs)[:,1] - np.array(outputs)[:,5])[i]/(np.array(outputs)[:,1])[i]
    for k in range(5):
            recent_error_list.append([error[counter - k], error2[counter - k], error3[counter - k], error4[counter - k]])  
    recent_error_list = np.array(recent_error_list)
    
    min_quality = np.min([1 - np.sum(recent_error_list.T[0]), 1 - np.sum(recent_error_list.T[1]), 1 - np.sum(recent_error_list.T[2]), 1 - np.sum(recent_error_list.T[3])])           
    
    quality.append([(1 - np.sum(recent_error_list.T[0])), (1 - np.sum(recent_error_list.T[1])), min_quality])
    residues.append([np.sum(recent_error_list.T[0])**2,np.sum(recent_error_list.T[1])**2,np.sum(recent_error_list.T[2])**2,np.sum(recent_error_list.T[3])**2])
#    if t > 2*nxt:
#        if np.sum(recent_error_list.T[0])**2 <= 0.15:
#            break
#        nxt += T
    recent_error_list = []
    counter += 1
    
    quality_list.append((1 - error_sum))                
    e_2 = e_1
    e_1 = e
    er = ysp - y
    
    eb_2 = eb_1
    eb_1 = eb
    eb = ysp2 - z
    
    u =u_1 + Kc*((er-e_1) + (T/tau_i)*er + (tau_d/T)*(er - 2*e_1 + e_2)) 
    v =v_1 + Kcb*((eb-eb_1) + (T/taub_i)*eb + (taub_d/T)*(eb - 2*eb_1 + eb_2)) 
    
    alp = u + dist
    bet = v + dist2
    
    y_2 = y_1
    y_1 = y
    z_2 = z_1
    z_1 = z
    alp_2 = alp_1
    alp_1 = alp
    bet_2 = bet_1
    bet_1 = bet
    u_2 = u_1
    u_1 = u
    v_2 = v_1
    v_1 = v
    yk_2 = yk_1
    yk_1 = yk
    zk_2 = zk_1
    zk_1 = zk   
    m_2 = m_1
    m_1 = m
    n_2 = n_1
    n_1 = n
      
    y = np.dot([[y_1, y_2, alp_1, alp_2, bet_1, bet_2]],[[a],[b],[c],[d],[eq],[f]])
    z = np.dot([[z_1, z_2, alp_1, alp_2, bet_1, bet_2]],[[aa],[bb],[cc],[dd],[ee],[ff]])
    y = y[0,0] + noise
    z = z[0,0] + noise2
    yk = np.dot([[yk_1, yk_2, alp_1, alp_2, bet_1, bet_2]], q_sampled)
    yk = yk[0,0]
    zk = np.dot([[zk_1, zk_2, alp_1, alp_2, bet_1, bet_2]], q_sampled2)  
    zk = zk[0,0] 
    m = np.dot([m_1, m_2, alp_1, alp_2, bet_1, bet_2], para_offlineA.T)
    n = np.dot([n_1, n_2, alp_1, alp_2, bet_1, bet_2], para_offlineB.T)
    
    #error = abs(yk - y)
    #error_list.append(error)
    phi_T = []
    y_list = []
    phi2_T = []
    z_list = []
#    tlist.append(t)
#    ilist.append(o)
    #print t    
#plot.plot(ilist, tlist, 'k', linewidth = 2.0)
#plot.xlabel("Time Window", fontsize = 20)
#plot.ylabel("Convergence Time", fontsize = 20)

my_sumA_inv = np.linalg.inv(my_sumA_1)
parameters = np.dot(my_sumA_inv, my_sumA_2)  

my_sumB_inv = np.linalg.inv(my_sumB_1)
parametersB = np.dot(my_sumB_inv, my_sumB_2) 
#
##print parameters
##print parametersB  
#
writefile(parameters, "off_para1.csv")
writefile(parametersB, "off_para2.csv")
#
#outputs = np.array(outputs)
#inputs = np.array(inputs)
##para_real = np.array(para_real)
##para_estim = np.array(para_estim)
##
light_online = 1.0
light_offline = 1.0
residues = np.array(residues)
plot.subplot(4,1,2)
#
plot.plot(tspan, outputs[:,1],'k',linewidth = 1.0)
plot.plot(tspan, outputs[:,3], 'k',label = "$z_{online}$", alpha = light_online, linewidth = 1.0)
plot.plot(tspan, outputs[:,5], 'k', label = "$z_{offline}$", alpha = light_offline)
plot.ylabel("$Output_2$", fontsize = 20)

plot.subplot(4,1,1)
plot.plot(tspan, outputs[:,0],'k', label = "$y$", linewidth = 1.0)
plot.plot(tspan, outputs[:,2], 'k', label = "$y_{online}$", alpha = light_online, linewidth = 1.0)
#
plot.plot(tspan, outputs[:,4], 'k',linewidth = 1.0, label = "$y_{offline}$", alpha = light_offline)
##plot.plot(tspan, outputs[:,5], label = "$z_{offline}$", alpha = light_offline)
plot.ylabel("$Output_1$", fontsize = 20)
##plot.legend(loc = 4)
#plot.subplot(4,1,3)
#plot.plot(tspan, inputs[:,0],'k', linewidth = 2.0)
#plot.ylabel("$Input_1$", fontsize = 20)
#plot.subplot(4,1,4)
#plot.plot(tspan, inputs[:,1], 'k', label = "$Controller$ $Output_2$", linewidth = 2.0)
#plot.ylabel("$Input_2$", fontsize = 20)
#plot.legend(loc = 4)
#
#
plot.subplot(4,1,3)
plot.plot(tspan, para_estim2, 'k', linewidth = 2.0)
plot.plot(tspan, para_real2, 'k--', linewidth = 2.0)
plot.ylabel("parameters($y_2$)", fontsize = 20)
plot.ylim([-0.67,1.5])
#
#
plot.subplot(4,1,4)
plot.plot(tspan, para_estim, 'k', linewidth = 2.0)
plot.plot(tspan, para_real, 'k--', linewidth = 2.0)
#
plot.ylabel("parameters($y_1$)", fontsize = 20)
plot.ylim([-0.67,1.5])
#plot.ylim([-0.5,1.43])
#plot.subplot(5,1,5)
#
#
#
#plot.plot(tspan, residues[:,0],'k',label = "Error in Output_1",linewidth = 2.0)
#plot.plot(tspan, residues[:,2],'k--',label = "Error in Output_2",linewidth = 2.0)
#plot.plot(tspan, residues[:,1],'k--',label = "Error in Output_2",linewidth = 2.0)
#plot.ylabel("Output Error", fontsize = 20)
#plot.ylim([0,1])
##plot.legend()
plot.xlabel("Time", fontsize = 20)

