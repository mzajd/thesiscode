#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:10:55 2020

@author: mdar
"""

from scipy.integrate import solve_ivp
#from scipy.optimize import fsolve
import math 
from scipy.integrate import solve_bvp
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt


def system(t, w, *p):
    '''
    Differential equations for the biofilm reactor model
    
    Arguments:
        
        w : vector of state variables
        w = [S, u, z]
        
        t: time
        p: vector of parameters
        p = [D, Sin, ADcV, gamV, Dku, Ap, E, gamDcp, kz, aAp, a, switch]
        
    '''
    S, u, z = w
    #D, Sin, ADcV, gamV, Dku, Ap, E, gamDcp, kz, aAp, a, switch = p
    D, Sin, gamma, A, Dc, V, ku, rho, E, a, kz, switch = p
    flux = jfun(S, z, switch)
    muuS = muu(S)
    
    # D*(Sin - S) - (A*Dc*flux + u*muuS/gamma)/V
    # u*(muuS - D - ku) + A*rho*E*z*z - a*u
    # gamma*Dc*flux/rho - z*kz - E*z*z + a*u/(A*rho)
        
    f = [D*(Sin - S) - (u*muuS/gamma + A*flux)/V,
         u*(muuS - D - ku) + A*rho*E*z*z - a*u,
             gamma*flux/rho - z*kz - E*z*z + a*u/(A*rho)
             ]
        
    return f

#load the MLP and NN objects
    
def load_pckl_file(filename):
    file = open(filename, 'rb' )
    pckl_obj = pickle.load(file)
    file.close()
    return pckl_obj

        
#NN_pckl = load_pckl_file('KNN_adaptive_interp.pkl')
NN_pckl = load_pckl_file('NN_adaptive_pckg_interp.pkl')

regr_pckl = load_pckl_file('MLPRegressor.pkl')

Xmean_pckl = load_pckl_file('X_mean.pkl')

std_pckl = load_pckl_file('std.pkl')


# function to calculate non dimensional numerical flux
def numerical_flux(k1, k2):
    def fun(x, y):
        for i in range(0, y[0].size):
            if y[0][i] < 0:
                y[0][i] = 0

        return np.vstack((y[1], k1 * y[0] / (k2 + y[0])))

# a is left endpoint, b is right endpoints
# index is the order of the derivative 
    def bc(ya, yb):
        # these are the boundary conditions
        return np.array([ya[1], yb[0] - 1])


    x = np.linspace(0, 1, 100)
    y = np.ones((2, x.size))

    return solve_bvp(fun, bc, x, y)

# functions to convert between k1, k2 and S, z
def k1_from_Sz(S, z):
    return z*z*rho*mumax/(gamma*Dc*S)
    
def k2_from_S(S):
    return K/S

def Sz_from_k1k2(k1, k2):
    if k2 == 0:
        return 0,0
    S = K/k2
    z = math.sqrt(k1*gamma*Dc*S/(rho*mumax))
    return S, z


def Rittman_flux(S, b):
    #nondimensionalize 
    # S has nan values fpr some b values
  
    S_b_nondim = S/Kz
    k = mumax/gamma
    S_min_nondim = b/(mumax - b)
    
    S_s_nondim = S_b_nondim

    def alpha(Smin):
        if Smin > 0.0001 and Smin <1:
            return 1.5739 + 0.32075*(-math.log(S_min_nondim)**0.15213)
        if Smin > 1 and Smin< 1000:
            return 1.5739 - 0.37149*(math.log(S_min_nondim)**0.31344)
        
    def beta(Smin):
        if Smin > 0.0001 and Smin <1:
            return 0.5014 + 0.01985*(-math.log(S_min_nondim)**0.19476)
        if Smin > 1 and Smin< 1000:
            return 0.5014 - 0.02726*(math.log(S_min_nondim)**0.52256)
    beta_par = beta(S_min_nondim)
   # print('Smin %f'%(S_min_nondim))
   # print('beta %f'%(beta_par))

    alpha_par = alpha(S_min_nondim)
    J_deep_nondim = (2*(S_s_nondim - math.log(1 + S_s_nondim)))**0.5
   
    S_diff = S_s_nondim/S_min_nondim-1
    if S_diff < 0:
        return 0
    J_nondim = J_deep_nondim*math.tanh(alpha_par*(S_diff)**beta_par)
    J = J_nondim*(Kz*k*rho*Dc)**0.5
    return J

def jfun(S, z, switch):
    ''' calculates dimensionalized flux value C'(lambda)
    S: substrate concentration
    z : biofilm thickness
    switch: 0, 1, or 2
    0 - uses solve_bvp 
    1 - uses analytical approx from Abbas paper
    2 - uses nearest neighbour interpolator
    3 - uses machine learning MLP model 
    4 - uses analytical flux with pen depth
    5 - J2
    6 - J1
    7-  Rittman flux
    '''
    if z <=0 or S <= 0:
        return 0

    # use parameter values and initial conditions to find k1 k2
    k1 = k1_from_Sz(S, z)
    # half saturation coefficient is 4
    k2 = k2_from_S(S)
    
    
               
    if switch == 0 :
    
        solution = numerical_flux(k1, k2)
        sol_num = solution.y[1][-1]*S/z
        
        flux = sol_num
     
    
    elif switch == 2:
        
        #flux = NN_pckl.predict([[k1, k2]])*S/z
        flux = NN_pckl(k1, k2)*S/z

    elif switch == 3:
        
        #normalize the input
        k1 -= Xmean_pckl[0]
        k1 /= std_pckl[0]
        k2 -= Xmean_pckl[1]
        k2 /= std_pckl[1]
        
        #reshape array to match input for the predict function
        k1k2_arr = np.array([k1, k2])
        k1k2_arr = np.reshape(k1k2_arr, (1, -1))  
        
        flux = regr_pckl.predict(k1k2_arr)*S/z
    
    elif switch == 1:
     
         flux =  mumax*rho*z*S/((gamma*Dc)*(Kz+ S))
    elif switch == 4:
         zp = math.sqrt(Dc*(gamma*(Kz + S0))/(mumax*rho))
   
         flux =  mumax*rho*min(z, zp)*S/((gamma*Dc)*(Kz+ S))
     
    # J2       
    elif switch == 5:
        theta = math.sqrt(mumax*rho/(Dc*gamma*(Kz+S)))
        flux = S*theta*np.tanh(theta*z)
        
    #J1
    elif switch == 6:
        theta = math.sqrt(mumax*rho/(Dc*gamma*Kz))
    
        flux = S*theta*np.tanh(theta*z)
        
    elif switch ==7 :
        #specify which loss value to use here
        return Rittman_flux(S, 0.2847)
      
    if flux <= 0:
        
         return 0
    else:
        return flux*Dc
     
    
        
     
   
        
        

def muu(S):
    return muumax*S/(Ku + S)

# initial values
# S0 can range from 0.1- 10
S0 = 20
# u normally 0.00001
u0 = 0.00001
# z normally 0.0001
z0 = 0.0001

w0 = [S0, u0, z0]
   
# parameters
mumax = 6
muumax = 6
# half saturation constants
# suspended bacteria
Ku = 4
K = 4
# biofilm
Kz = 4
# dilution rate varies from 0.42-85
D = 3
#substrate concentration at inlet
Sin = S0
# volume
V = 0.00118
# area
A = 0.055
# diffusion coefficient
Dc = 0.0001
# biomass density
rho = 10000
# cell death rates
kz = 0.4
ku = 0.4

# attachment rate normally 1
a = 1
# erosion/detachment varies 22.8-1000 normally 22.8
E = 22.8
# yield
gamma = 0.63

# controls which flux approx is used
switch = 0


if __name__ == '__main__':
   S0_arr = [0.2, 0.3,0.5, 1.0, 2.0, 3.0, 4.0, 10.0,40.0, 80.0, 175.0, 250.0]

   #S0_arr = [0.7, 1.0, 2.0, 3.0, 4.0, 10.0,40.0, 80.0, 175.0, 250.0]
   #S0_arr = [4, 10, 20, 30, 40, 45, 50, 65, 80, 100, 125, 150, 175, 200, 225, 250]
   #S0_arr = [4, 10, 20, 40 ,80]
    
   def rel_err(true, approx):
       if true ==0:
           return approx/(true + 1E-26)
       return (true- approx)/(true)
    
    # to print out the tables for each flux comparing S, u, z, flux, time for eqbm
   # using latex format
    
   def print_Suzjtime_S0(S0_arr, flux): 
        '''
        flux follows the same switches as jfun
        0 num, 1 alg, 2 nn, 3 mlp, 4 pen depth, 5 j2, 6 j1
        ''' 
        approx_sol_arr = []
        num_sol_arr = []
        num_time_arr = []
        approx_time_arr = []
        for Sinit in S0_arr:
            p = (D, Sinit, gamma, A, Dc, V, ku, rho, E, a, kz, flux)
            atime = time.time()
            approx_sol_arr.append(solve_ivp(system, [0, 100], [Sinit, u0, z0], method="LSODA",
                         args=(p), dense_output =True))
            atime = time.time() - atime
            approx_time_arr.append(atime)
            
            ntime = time.time()
            pnum = (D, Sinit, gamma, A, Dc, V, ku, rho, E, a, kz, 0)
            num_sol_arr.append(solve_ivp(system, [0, 100], [Sinit, u0, z0], method="LSODA",
                         args=(pnum), dense_output =True))
            ntime = time.time() - ntime
            num_time_arr.append(ntime)
        
        for i in range(len(S0_arr)):
            S_in = S0_arr[i]
            numsol = num_sol_arr[i]
            approxsol = approx_sol_arr[i]
            
            Snum_eq = numsol.y[0][-1]
            Sapprox_eq =  approxsol.y[0][-1]
            S_err = rel_err(Snum_eq, Sapprox_eq)
            
            unum_eq = numsol.y[1][-1]
            uapprox_eq = approxsol.y[1][-1]
            u_err = rel_err(unum_eq, uapprox_eq)
            
            znum_eq = numsol.y[2][-1]
            zapprox_eq = approxsol.y[2][-1]
            z_err = rel_err(znum_eq, zapprox_eq)
                   
            time_num = num_time_arr[i]
            time_approx = approx_time_arr[i]
            time_err = rel_err(time_num, time_approx)
            
            fluxnum_eq = jfun(Snum_eq, znum_eq, 0)
            fluxapprox_eq = jfun(Sapprox_eq, zapprox_eq, flux)
            flux_err = rel_err(fluxnum_eq, fluxapprox_eq)
     
            
            
            print("%.1f & %.2E & %.2E & %.2E & %.2E & %.2E\n"%(S_in,Snum_eq,\
                                                             Sapprox_eq,\
                                                            S_err, unum_eq,\
                                                            uapprox_eq))
            print("& %.2E & %.2E & %.2E & %.2E & %.2E\n"%(u_err, znum_eq,\
                                                          zapprox_eq, z_err,\
                                                          time_num))
            print("& %.2E & %.2E & %.2E & %.2E & %.2E\n"%(time_approx,time_err,\
                                                          fluxnum_eq,  fluxapprox_eq,\
                                                          flux_err))
            print("\\\ \n \hline\n")

# test run of reactor code        
p = (D, S0, gamma, A, Dc, V, ku, rho, E, a, kz, 7)
solution_num = solve_ivp(system, [0, 100], w0, method="LSODA",
             args=(p), dense_output =True)

print("S: %.4E u: %.4E z: %.4E"%(solution_num.y[0][-1], solution_num.y[1][-1], solution_num.y[2][-1]))
            

# flux comparison file generator

def make_Szj_file(flux):
    '''creates file  lambda S k1 k2 fluxnum fluxapprox for plotting'''
    if flux == 2:
        Szj_file = open("Szj_file_NN.txt", 'w+')
    if flux == 3:
        Szj_file = open("Szj_file_MLP.txt", 'w+')
    if flux == 7:
        Szj_file = open("Szj_file_Ritt.txt", 'w+')
    
    for k1 in np.arange(0, 100):
        for k2 in np.arange(0, 100):
            S, z = Sz_from_k1k2(k1, k2)
            j1 = jfun(S,z, 0) 
            j2 = jfun(S, z, flux)
            Szj_file.write("%.5f %.5f %.5f %f %.8f %.8f\n"%(z, S, k1, k2, j1, j2))
        Szj_file.write("\n")
        
    Szj_file.close()


#plot k1 and k2 as functions of time
    
def plot_k1k2(sol):
     plt.plot(sol.t, k1_from_Sz(sol.y[0], sol.y[2]), 'r', label="k1")
     plt.xlabel('t', fontsize =15)
     plt.ylabel('k1', fontsize =15)
     plt.show()
     
     plt.plot(sol.t, k2_from_S(sol.y[0]), 'r', label="k2")
     plt.xlabel('t', fontsize =15)
     plt.ylabel('k2', fontsize =15)
     plt.show()
     

def plot_transients_vs_num(flux, S_start):
    p_num = (D, S0, gamma, A, Dc, V, ku, rho, E, a, kz, 0)
    p_flux = (D, S0, gamma, A, Dc, V, ku, rho, E, a, kz, flux)
    
    sol_num = solve_ivp(system, [0, 100], [S_start, u0, z0], method="LSODA",
             args=(p_num), dense_output =True)
    sol_flux = solve_ivp(system, [0, 100], [S_start, u0, z0], method="LSODA",
             args=(p_flux), dense_output =True)
    
    plt.plot(sol_num.t, sol_num.y[0], 'r', label="numerical")
    plt.plot(sol_flux.t, sol_flux.y[0], 'b', linestyle='dashed',label="approx")
    plt.xlabel('t', fontsize =15)
    plt.ylabel('S', fontsize =15)
    plt.xlim([-1,20])
    plt.legend()
    plt.show()
    
    plt.plot(sol_num.t, sol_num.y[1], 'r', label="numerical")
    plt.plot(sol_flux.t, sol_flux.y[1], 'b', linestyle='dashed',label="approx")
    plt.xlabel('t', fontsize =15)
    plt.ylabel('u', fontsize =15)
    plt.xlim([-1,20])
    plt.legend()
    plt.show()
    
    plt.plot(sol_num.t, sol_num.y[2], 'r', label="numerical")
    plt.plot(sol_flux.t, sol_flux.y[2], 'b', linestyle='dashed',label="approx")
    plt.xlabel('t', fontsize =15)
    plt.ylabel('z', fontsize =15)
    plt.xlim([-1,20])
    plt.legend()
    plt.show()
    
def Rittman_flux_vs_b(S, z, b_array):
    R = []
    num = []
    for b in b_array:
        R.append(Rittman_flux(S, b))
        num.append(jfun(S, z, 0))
        
    plt.plot(b_array, R,label="rittman")
    plt.plot(b_array, num, label="numerical")
    plt.xlabel('kz', fontsize =15)
    plt.ylabel('flux', fontsize =15)
    plt.legend()
    plt.xlim([3,6])
    plt.show()

#Rittman_flux_vs_b(4, np.linspace(0.0006,2.999,20))
#Rittman_flux_vs_b(236, np.linspace(3.0001,5.9,20))
    
def plot_all_approx_transients(S, var):
    p_num = (D, S, gamma, A, Dc, V, ku, rho, E, a, kz, 0)
    p_mlp = (D, S, gamma, A, Dc, V, ku, rho, E, a, kz, 3)
    p_pen_dep = (D, S, gamma, A, Dc, V, ku, rho, E, a, kz, 4)
    p_j2 =(D, S, gamma, A, Dc, V, ku, rho, E, a, kz, 5)
    p_j1 =(D, S, gamma, A, Dc, V, ku, rho, E, a, kz, 6)
    p_ritt = (D, S, gamma, A, Dc, V, ku, rho, E, a, kz, 7)
    
    sol_num = solve_ivp(system, [0, 100], [S, u0, z0], method="LSODA",
             args=(p_num), dense_output =True)
    sol_mlp = solve_ivp(system, [0, 100], [S, u0, z0], method="LSODA",
             args=(p_mlp), dense_output =True)
    sol_pen_dep = solve_ivp(system, [0, 100], [S, u0, z0], method="LSODA",
             args=(p_pen_dep), dense_output =True)
    sol_j2 = solve_ivp(system, [0, 100], [S, u0, z0], method="LSODA",
             args=(p_j2), dense_output =True)
    sol_j1 = solve_ivp(system, [0, 100], [S, u0, z0], method="LSODA",
             args=(p_j1), dense_output =True)
    sol_ritt = solve_ivp(system, [0, 100], [S, u0, z0], method="LSODA",
             args=(p_ritt), dense_output =True)
    
    plt.plot(sol_num.t, sol_num.y[var], 'r', label="numerical")
    plt.plot(sol_mlp.t, sol_mlp.y[var], 'b', linestyle='dashed',label="mlp")
    plt.plot(sol_pen_dep.t, sol_pen_dep.y[var], 'y', label="alg pen depth")
    plt.plot(sol_j2.t, sol_j2.y[var], 'g', linestyle='dashed',label="j2")
    plt.plot(sol_j1.t, sol_j1.y[var], 'r', linestyle='dashed',label="j1")
    plt.plot(sol_ritt.t, sol_ritt.y[var], 'b', linestyle='dotted',label="rittman")
    
    if var == 0:
        plt.xlabel('t', fontsize =15)
        plt.ylabel('S', fontsize =15)
    if var == 1:
        plt.xlabel('t', fontsize =15)
        plt.ylabel('u', fontsize =15)
    if var == 2:
        plt.xlabel('t', fontsize =15)
        plt.ylabel('z', fontsize =15)
        
    
    plt.xlim([-1,10])
    plt.legend()
    plt.show()


   
            