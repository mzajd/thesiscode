#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:54:42 2020

@author: mdar

generates a grid with given relative error
creates a file with k1 k2 and flux for the given grid called
adapgridflux.txt

"""

import solveroneinstance_time as sol

import numpy as np
from scipy.integrate import solve_bvp
k1 = np.array([0.1,  100])
k2 = np.array([0.1, 100])


def checkmaxerrcell(x1, x2, y1, y2):
    '''checks the max difference between 4 points of a grid cell'''
    flux = []
    maxerr = 0
    for i in [x1, x2]:
        for j in [y1, y2]:
            sol.k1 = i
            sol.k2 = j
    
    
            solution = sol.solve_bvp(sol.fun, sol.bc, sol.x, sol.y)
            J = solution.yp[0][-1]
            flux.append(J)
                     
    for i in range(0, len(flux)-1):
        for j in range(i+1, len(flux)):
            maxerr = max(maxerr, abs(flux[j]-flux[i])*200/abs(flux[j]+ flux[i]))
            
    return maxerr

 

def adaptivegrid(x, y, error):
    '''x - array gridpoints
        y - array gridpoints
        error - refines grid until error is below this val
        each time error is not low enough another meshpoint is added to entire
        grid'''
        
    currerr = checkmaxerrcell(x[0], x[1], y[0], y[1])
    
    if currerr > error:
        #print("%f\n"%(currerr))
        x = np.append(x, (x[1]+x[0])/2)
        #print("new x %f\n"%((x[1]+x[0])/2))
        y = np.append(y, (y[1]+y[0])/2)
        
        newx, newy = adaptivegrid([x[0],x[2]], [y[0],y[2]], error)
        x = np.append(x, newx)
        y = np.append(y, newy)
        
        newx, newy = adaptivegrid([x[0],x[2]], [y[2],y[1]], error)
        x = np.append(x, newx)
        y = np.append(y, newy)
        
        newx, newy = adaptivegrid([x[2],x[1]], [y[2],y[1]], error)
        x = np.append(x, newx)
        y = np.append(y, newy)
        
        newx, newy = adaptivegrid([x[2],x[1]], [y[0],y[2]], error)
        x = np.append(x, newx)
        y = np.append(y, newy)
        
        
    return x,y

xK1, yK2 = adaptivegrid(k1, k2, 50)

# removes duplicates in x and y  
xK1 = np.unique(xK1)
yK2 = np.unique(yK2)

# creates file and writes the grid values to it 
file = open("./adapgridflux_50.txt", "w+")

def num_sol(k1s, k2s):
    
    # stacks two first order equations to solve
    def fun(x, y):
        for i in range(0, y[0].size):
            if y[0][i] < 0:
                y[0][i] = 0
    
        return np.vstack((y[1], k1s * y[0] / (k2s + y[0])))
    
    # a is left endpoint, b is right endpoints
    # index is the order of the derivative 
    def bc(ya, yb):
        return np.array([ya[1], yb[0] - 1])
    
    
    x = np.linspace(0, 1, 100)
    y = np.ones((2, x.size))
    
    return solve_bvp(fun, bc, x, y)

for k1 in xK1:
    for k2 in yK2:
       # sol.k1 = k1
       # sol.k2 = k2
       # solution = sol.solve_bvp(sol.fun, sol.bc, sol.x, sol.y)
       solution = num_sol(k1, k2)
       J = solution.yp[0][-1] 
       
       file.write("%.17f %.17f %.17f\n"%(k1, k2, J))


       
       
file.close()
        
                    
        
       
                
                    
                    
