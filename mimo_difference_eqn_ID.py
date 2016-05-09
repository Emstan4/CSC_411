# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:39:05 2016

@author: charles
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plot
import scipy
import csv

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
tend = 500
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
e = a_3
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
dist = 0.0
dist2 = 0

my_sumA_1 = np.zeros((6,6))
my_sumA_2 = np.zeros((6,1))
my_sumB_1 = np.zeros((6,6))
my_sumB_2 = np.zeros((6,1))

for i, t in enumerate(tspan):
    
    noise = sigma*np.random.rand()
    noise2 = sigma*np.random.rand()
    
    outputs[i] = [y, z, yk, zk, m, n]
    inputs.append([ysp, ysp2, alp, bet])
    para_estim[i] = Q_0.T[0]
    para_estim2[i] = Q2_0.T[0]
    para_real[i] = [a,b,c,d,e,f]
    para_real2[i] = [aa,bb,cc,dd,ee,ff]            
    ysp = step(0.7,0.0,300,t)
    ysp2 = step(0.1,0.0,300,t) 
    
    #PRBS of the disturbance
    s = np.random.randint(10,30)
    s2 = np.random.randint(10,30)
    dist = 0.02*square_wave(s,t)
    dist2 =0.02*square_wave(s2,t)
    
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
        error_sum = np.sum(error_list)
#        if (1 - error_sum) >= 0.85:
#            break
        
        error_list = []
        q_sampled = Q_t
        q_sampled2 = Q2_t
        
        next_time2 += 20
    quality_list.append((1 - error_sum))                
    e_2 = e_1
    e_1 = e
    er = ysp - y
    
    eb_2 = eb_1
    eb_1 = eb
    eb = ysp2 - z
    
    u = u_1 + Kc*((er-e_1) + (T/tau_i)*er + (tau_d/T)*(er - 2*e_1 + e_2)) 
    v = v_1 + Kcb*((eb-eb_1) + (T/taub_i)*eb + (taub_d/T)*(eb - 2*eb_1 + eb_2)) 
    
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
      
    y = np.dot([[y_1, y_2, alp_1, alp_2, bet_1, bet_2]],[[a],[b],[c],[d],[e],[f]])
    z = np.dot([[z_1, z_2, alp_1, alp_2, bet_1, bet_2]],[[aa],[bb],[cc],[dd],[ee],[ff]])
    y = y[0,0] + noise
    z = z[0,0] + noise2
    yk = np.dot([[yk_1, yk_2, alp_1, alp_2, bet_1, bet_2]], q_sampled)
    yk = yk[0,0]
    zk = np.dot([[zk_1, zk_2, alp_1, alp_2, bet_1, bet_2]], q_sampled2)  
    zk = zk[0,0] 
    m = np.dot([m_1, m_2, alp_1, alp_2, bet_1, bet_2], para_offlineA.T)
    n = np.dot([n_1, n_2, alp_1, alp_2, bet_1, bet_2], para_offlineB.T)
    
    error = abs(yk - y)
    error_list.append(error)
    phi_T = []
    y_list = []
    phi2_T = []
    z_list = []
#print t    
#print q_sampled.T
#print np.array([[a],[b],[c],[d],[e],[f]]).T   
#plot.plot(quality_list) 
my_sumA_inv = np.linalg.inv(my_sumA_1)
parameters = np.dot(my_sumA_inv, my_sumA_2)  

my_sumB_inv = np.linalg.inv(my_sumB_1)
parametersB = np.dot(my_sumB_inv, my_sumB_2) 

    
writefile(parameters, "off_para1.csv")
writefile(parametersB, "off_para2.csv")

outputs = np.array(outputs)
inputs = np.array(inputs)
para_real = np.array(para_real)
para_estim = np.array(para_estim)
plot.subplot(2,2,1)
plot.plot(tspan, outputs[:,0], label = "$y$")
plot.plot(tspan, outputs[:,1], label = "$z$")
plot.plot(tspan, outputs[:,2], label = "$y_{predicted}$", alpha = 0.5)
plot.plot(tspan, outputs[:,3], label = "$z_{predicted}$", alpha = 0.5)
plot.plot(tspan, outputs[:,4], label = "$z_{predicted}$")
plot.plot(tspan, outputs[:,5], label = "$z_{predicted}$")
plot.ylabel("outputs")
#plot.legend(loc = "best")
plot.subplot(2,2,2)
plot.plot(tspan, inputs)
#plot.plot(tspan, inputs[:,0], label = "$setpoint_y$")
#plot.plot(tspan, inputs[:,1], label = "$setpoint_z$")
plot.ylabel("setpoints")
plot.subplot(2,2,3)
plot.plot(tspan, para_estim2, 'r')
plot.plot(tspan, para_real2, 'k')
plot.ylabel("parameters(z)")
plot.subplot(2,2,4)
plot.plot(tspan, para_estim, 'b')
plot.plot(tspan, para_real, 'k')
plot.xlabel("time")
plot.ylabel("parameters(y)")

#plot.plot(tspan, quality_list)
#plot.ylim([0,1])
#plot.show()
