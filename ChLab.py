# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:08:11 2015

@author: Charles
"""
from __future__ import division
import numpy as np
import csv
from matplotlib import pyplot as plot
import math as ms

# Storing Function                 
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


#STEP DISTURBANCE FUNCTION
def step(start, stepsize, tstep, t):
    if t <= tstep:
        return start
    else:
        return start + stepsize 

def poly_order(M):
    for i in range(len(M)):
        count = M[i]
        if count != 0:
            break
    m_order = (len(M)-1) - i    
    return m_order   

#Pade approximation function with specified order
def pade(numerator_order, denominator_order, Dtime):
    n = denominator_order
    m = numerator_order
    my_list = []
    my_list2 = []
    for i in range(m+1):
        Pm = ((ms.factorial(m+n-i)*ms.factorial(m))/(ms.factorial(m+n)*ms.factorial(i)*ms.factorial(m-i)))*(-1*Dtime)**i
        my_list.append(Pm)
    
    for i in range(n+1):
        Pm2 = ((ms.factorial(m+n-i)*ms.factorial(n))/(ms.factorial(m+n)*ms.factorial(i)*ms.factorial(n-i)))*(1*Dtime)**i
        my_list2.append(Pm2)
    
    listA = np.array(my_list)
    listB = np.array(my_list2)
    list1 = []
    list2 = []
    for k in range(len(listA)):
        list1.append(listA[-k-1])
    for k in range(len(listB)):
        list2.append(listB[-k-1])
    return [list1, list2]
    
#Converts a transfer function to a state-space representation    
def state_space(numerator,denominator):
    G = numerator
    F = denominator
    numer2 = G
    denom2 = F
    def multi(numer):
        a,b = np.shape(numer)
        N =[] 
        
        for i in range(len(numer)):
            f = []
            for k in range(len(numer[0])):
                
                f.append((numer[i][-k-1]))
        
            N.append(f)
        for i in range(len(N)-1):
            mul = np.polynomial.polynomial.polymul(N[0], N[i+1])
            N[0] = mul
        r = []
        
        for i in range(len(N[0])):
            r.append(N[0][-i-1])
        return r
        
    numerB = multi(numer2)
    denomB = multi(denom2)
    
    denomX = denomB
    numerX = multi2(numerB, denomB)
    G = numerX
    F = denomX
    N = []
    for i in range(len(F)):
        s = F[i]/F[0]
        N.append(s)
    M = []
    for i in range(len(G)):
        si = G[i]/F[0]
        M.append(si)    
    m_order = poly_order(M)
    n_order = poly_order(N)
     
    if m_order == 0 and n_order == 0:
        A = [[0]]
        B = [[0]]
        C = [[0]]
        D = G/F
    else:
        A = np.zeros((n_order,n_order))
        B = np.zeros((n_order,1))
        B[len(B) - 1,0] = 1
        C = np.zeros((1,n_order))
        D = np.zeros((1,1))
        D[0,0] = M[0]

        for i in range(len(A)):
            A[i-1,i] = 1
            A[len(A)-1,i] = -N[-i-1]
            C[0,i] = M[0]*(-N[-i-1]) + M[-i-1]                     
    return np.matrix(A), np.matrix(B),np.matrix(C),np.matrix(D)


#Plots a response given a transfer function.
def plotter(numerator, denominator,m,n,t_end, initial_input, stepsize, t_step, Dtime,sampling_interval):
    delta = sampling_interval    
    numer2 = numerator
    denom2 = denominator
    pad = pade(m,n,Dtime)
    Pnum = pad[0]
    Pdeno = pad[1]
    
    def multi(numer):
        a,b = np.shape(numer)
        N =[] 
        
        for i in range(len(numer)):
            f = []
            for k in range(len(numer[0])):
                
                f.append((numer[i][-k-1]))
        
            N.append(f)
        for i in range(len(N)-1):
            mul = np.polynomial.polynomial.polymul(N[0], N[i+1])
            N[0] = mul
        r = []
        
        for i in range(len(N[0])):
            r.append(N[0][-i-1])
        return r
    numerB = multi(numer2)
    denomB = multi(denom2)
    
    if len(numerB) > len(denomB):
        print "error"
    else:
        def multi2(numerA, denomA):
            t = np.zeros(len(denomA))
            diff = len(denomA) - len(numerA)
            for i in range(len(numerA)):
                t[i+diff] = numerA[i]
            return t
        if len(Pdeno) > len(denomB):
            denomC = multi2(denomB, Pdeno)
            numerC = multi2(numerB, Pdeno)
            Pnumer = multi2(Pnum, Pdeno)
            Pdenom = Pdeno
        elif len(Pdeno) <= len(denomB):
            Pdenom = multi2(Pdeno, denomB)
            numerC = multi2(numerB, denomB)
            Pnumer = multi2(Pnum, denomB)
            denomC = denomB
            
        v = [] 
        for i in range(len(denomC)):
            v.append(denomC[i])
            
        denom = [Pdenom, v]                           
        w = [] 
        for i in range(len(denomC)):
            w.append(numerC[i])            
        numer = [Pnumer, w]                  
        denomX = multi(denom)
        numerX = multi2(multi(numer), denomX)
        
        NN = multi2(multi(numer2), multi(denom2))
        DD = multi(denom2)
        
        def tf(dt, t_start, t_end, initial_input, stepsize, t_step):
            A, B, C, D = state_space([numerX], [denomX])
            A1, B1, C1, D1 = state_space([NN], [DD])
            
            Nstates = A.shape[0]
            Nstates2 = A1.shape[0]
            
            z = np.zeros((Nstates, 1))
            z2 = np.zeros((Nstates2, 1))
            
            tspan = np.arange(t_start,t_end,dt)
            yplot = np.zeros(len(tspan))
            y2plot = np.zeros(len(tspan))
            y3plot = np.zeros(len(tspan))
            ymplot = np.zeros(len(tspan))
            lst=[]
            time=[]
            
            next_time=0
            ym = 0
            y2 = 0
            for i,t in enumerate(tspan):
                y3plot[i] = ym
                y2plot[i] = y2
                if t>= next_time:                
                    y_sam = np.interp(t-Dtime,tspan,y3plot)        
                    next_time += delta
                    lst.append(y_sam)
                    time.append(t)                
                e = step(initial_input, stepsize, t_step,t)
                
                dzdt = A*z + B*e
                y   = C*z + D*e
                
                dz2dt = A1*z2 + B1*e
                y2   = C1*z2 + D1*e
                
                z += dzdt*dt
                z2 += dz2dt*dt
                yplot[i] = y
                
                ym = np.interp(t - Dtime, tspan, y2plot)+ 0.01*np.random.randn()
                ymplot[i] = ym 
                
            AA=np.zeros((len(lst),2)) 
            for i in range(len(lst)):
                AA[i][1]=lst[i]
                AA[i][0]=time[i]
                
            plot.plot(tspan, ymplot,tspan, yplot)
            legend_list = ["Actual", "Approximated"]
            plot.legend(legend_list)
            plot.xlabel("Time")
            plot.ylabel("Y_output")
            plot.title("Response")
            plot.show()
            writefile(AA,'Sampled_Data.csv')
            print [lst]
            
        f = tf(0.001, 0, t_end, initial_input, stepsize, t_step)
    return f
   
def multi(numer):
        a,b = np.shape(numer)
        N =[] 
        
        for i in range(len(numer)):
            f = []
            for k in range(len(numer[0])):
                
                f.append((numer[i][-k-1]))
        
            N.append(f)
        for i in range(len(N)-1):
            mul = np.polynomial.polynomial.polymul(N[0], N[i+1])
            N[0] = mul
        r = []
        
        for i in range(len(N[0])):
            r.append(N[0][-i-1])
        return r
        
def multi2(numerA, denomA):
    t = np.zeros(len(denomA))
    diff = len(denomA) - len(numerA)
    for i in range(len(numerA)):
        t[i+diff] = numerA[i]
    return t
        
def bode(a, b, D, num_ord, den_ord, w_initial, w_final):
    p_num = pade(num_ord, den_ord, D)[0]
    p_den = pade(num_ord, den_ord, D)[1]
    
    P_DNT = []
    P_NMT = []
    for k in range(len(p_num)):
        P_NMT.append(p_num[-k-1])
    for k in range(len(p_den)):
        P_DNT.append(p_den[-k-1])
        
    p_rootsA = np.polynomial.polynomial.polyroots(P_DNT) 
    p_rootsB = np.polynomial.polynomial.polyroots(P_NMT)
    
    a = multi(a)    
    b = multi(b)

    DNT = []
    NMT = [] 
    for k in range(len(a)):
        NMT.append(a[-k-1])
    for k in range(len(b)):
        DNT.append(b[-k-1])
        
    roots = np.polynomial.polynomial.polyroots(DNT)#[-1/5,-1]
    roots2 = np.polynomial.polynomial.polyroots(NMT)
    lstA = []
    lstB = []
    dt = 0.01
    wspan = np.arange(w_initial,w_final,dt)
    mod = []
    for i in range(len(wspan)):
        cntA = 0+0j
        cnt1 = 0+0j
        cnt2 = 0+0j
        cntB = 0+0j
        cntC = 0+0j
        cntD = 0+0j
        w = wspan[i]
        for k in range(len(b)):
            DOWN = (b[k])*(w*1j)**(len(b)-1-k)
            cnt1+=DOWN
            
        for k in range(len(a)):
            UP = (a[k])*(w*1j)**(len(a)-1-k)    
            
            cnt2+=UP
        cnt = cnt2/cnt1
        res = (cnt.real**2 + cnt.imag**2)**0.5
        mod.append(res)
        
        for k in range(len(p_rootsB)):
            Y = 1j*w - p_rootsB[k]
            if p_rootsB[k] == 0:
                phi2 = +90
            else:
                phi2 = (np.arctan(Y.imag/Y.real))*(180/np.pi)
            cntD += phi2
            
        for k in range(len(p_rootsA)):
            tA = (1j*w - p_rootsA[k])
            
            if p_rootsA[k] == 0:
                phi = -90

            else:
                phi = -(np.arctan(tA.imag/tA.real))*(180/np.pi)
                    
            cntB += phi
            
        for z in range(len(roots2)):
            X = (1j*w - roots2[z])
            if roots2[z]==0:
                angleA = +90
            else:
                angleA = (np.arctan(X.imag/X.real))*(180/np.pi)
            cntC += angleA
            
        for q in range(len(roots)):
            T = (1j*w - roots[q])
            
            if roots[q] == 0:
                angleB = -90

            else:
                angleB = -(np.arctan(T.imag/T.real))*(180/np.pi)
                
            cntA += angleB 
            
        lstA.append(cntA + cntC - (D*w*180/np.pi))
        lstB.append(cntA + cntB + cntC + cntD)
        
    plot.subplot(2,1,1)
    plot.loglog(wspan, mod)
    plot.ylabel("$AR$", fontsize = 20)
    plot.title("$Bode$ $Plot$", fontsize = 30)
    plot.subplot(2,1,2)
    plot.plot(wspan, lstA, wspan, lstB)
    legend_list = ["Actual", "Approximated"]
    plot.legend(legend_list)
    plot.ylabel(r"$\phi$ $(deg)$", fontsize = 20)
    plot.xlabel(r"$\omega$", fontsize = 20)
    plot.xscale(u'log')
    plot.show()
        
    
    
def closedloop(setpoint,numerp,denomp,Dp, numerc,denomc,Dc, numerv,denomv,Dv, numerm,denomm,Dm, numersp,denomsp,Dsp, numerd,denomd,Dd, t_end, initial_input, stepsize, t_step):
                
    def multi(numer):
        a,b = np.shape(numer)
        N =[] 
        
        for i in range(len(numer)):
            f = []
            for k in range(len(numer[0])):
                
                f.append((numer[i][-k-1]))
        
            N.append(f)
        for i in range(len(N)-1):
            mul = np.polynomial.polynomial.polymul(N[0], N[i+1])
            N[0] = mul
        r = []
        
        for i in range(len(N[0])):
            r.append(N[0][-i-1])
        return r
    #Process
    numerB = multi(numerp)
    denomB = multi(denomp)
    #controller
    numerC = multi(numerc)
    denomC = multi(denomc)
    #Valve
    numerD = multi(numerv)
    denomD = multi(denomv)
    #Measured
    numerE = multi(numerm)
    denomE = multi(denomm)
    #Setpoint
    numerF = multi(numersp)
    denomF = multi(denomsp)
    #Disturbance
    numerG = multi(numerd)
    denomG = multi(denomd)
    
    if len(numerB) > len(denomB):
        print "error"
    else:
        def multi2(numerA, denomA):
            t = np.zeros(len(denomA))
            diff = len(denomA) - len(numerA)
            for i in range(len(numerA)):
                t[i+diff] = numerA[i]
            return t

        #Process
        num_p = multi2(numerB, denomB)
        den_p = multi(denomp)
        
        #controller
        num_c = multi2(numerC, denomC)
        den_c = multi(denomc)
        
        #valve
        num_v = multi2(numerD, denomD)
        den_v = multi(denomv)
        
        #measured
        num_m = multi2(numerE, denomE)
        den_m = multi(denomm)
        
        #setponit
        num_sp = multi2(numerF, denomF)
        den_sp = multi(denomsp)
        
        #disturbance
        num_d = multi2(numerG, denomG)
        den_d = multi(denomd)
        
        def tf(dt, t_start, t_end, initial_input, stepsize, t_step):
            
            tspan = np.arange(t_start,t_end,dt)
            
            #Process
            
            A1, B1, C1, D1 = state_space([num_p], [den_p])
            Nstates = A1.shape[0]
            z = np.zeros((Nstates, 1))                                    
            yplot = np.zeros(len(tspan))            
            ymplot = np.zeros(len(tspan))
            
            #controller
            
            A2, B2, C2, D2 = state_space([num_c], [den_c])
            Nstates2 = A2.shape[0]
            z2 = np.zeros((Nstates2, 1))                                    
            yplot2 = np.zeros(len(tspan))            
            ymplot2 = np.zeros(len(tspan))
            
            #valve
            
            A3, B3, C3, D3 = state_space([num_v], [den_v])
            Nstates3 = A3.shape[0]
            z3 = np.zeros((Nstates3, 1))                                    
            yplot3 = np.zeros(len(tspan))            
            ymplot3 = np.zeros(len(tspan))
            
            #measured
            
            A4, B4, C4, D4 = state_space([num_m], [den_m])
            Nstates4 = A4.shape[0]
            z4 = np.zeros((Nstates4, 1))                                    
            yplot4 = np.zeros(len(tspan))            
            ymplot4 = np.zeros(len(tspan))
            
            #setpoint
            
            A5, B5, C5, D5 = state_space([num_sp], [den_sp])
            Nstates5 = A5.shape[0]
            z5 = np.zeros((Nstates5, 1))                                    
            yplot5 = np.zeros(len(tspan))            
            
            #disturbance
            
            A6, B6, C6, D6 = state_space([num_d], [den_d])
            Nstates6 = A6.shape[0]
            z6 = np.zeros((Nstates6, 1))                                    
            yplot6 = np.zeros(len(tspan))            
            ymplot6 = np.zeros(len(tspan))
            output = np.zeros(len(tspan))
            
            y4 = 0
            for i,t in enumerate(tspan):
                
                sp = step(setpoint,0.0,0,t)
                e = (sp - y4)
                
                #Setpoint
                
                dzdt5 = A5*z5 + B5*sp
                y5   = C5*z5 + D5*sp
                z5 += dzdt5*dt
                yplot5[i] = y5
                
                
                #Controller
                
                dzdt2 = A2*z2 + B2*e
                y2   = C2*z2 + D2*e
                z2 += dzdt2*dt
                yplot2[i] = y2
                ym2 = np.interp(t - Dc, tspan, yplot2)
                
                #Valve

                dzdt3 = A3*z3 + B3*ym2
                y3   = C3*z3 + D3*ym2
                z3 += dzdt3*dt
                yplot3[i] = y3
                ym3 = np.interp(t - Dv, tspan, yplot3)
                
                
                #disturbance
                d = step(initial_input, stepsize, t_step,t)
                dzdt6 = A6*z6 + B6*d
                y6   = C6*z6 + D6*d
                z6 += dzdt6*dt
                yplot6[i] = y6
                ym6 = np.interp(t - Dd, tspan, yplot6)
                
                #Process
                
                dzdt = A1*z + B1*(ym3+ym6)
                y   = C1*z + D1*(ym3+ym6)
                z += dzdt*dt
                yplot[i] = y
                ym = np.interp(t - Dp, tspan, yplot)
                
                #Measured 
                
                dzdt4 = A4*z4 + B4*ym
                y4   = C4*z4 + D4*ym
                z4 += dzdt4*dt
                yplot4[i] = y4
                ym4 = np.interp(t - Dm, tspan, yplot4)
                
                ymplot4[i] = ym4
                output[i] = ym                                
                ymplot[i] = ym                                                
                ymplot6[i] = ym6
                ymplot3[i] = ym3                
                ymplot2[i] = ym2
                
            plot.plot(tspan, output,'k',tspan, yplot5)
            legend_list = ["Output", "Setpoint"]
            plot.legend(legend_list)
            plot.xlabel("Time")
            plot.ylabel("Y_output")
            plot.title("Closed loop response")
            plot.show()            
        f = tf(0.005, 0, t_end, initial_input, stepsize, t_step)
    return f


    

   
