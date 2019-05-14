# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:50:00 2019

@author: Tushar
"""

#crab sattelites
import pandas as pd
import numpy as np

#SET GLM update parameters
max_iter = 500
tol = 0.000001

#readcrab.xlsx and save in numpy array
xl_file = pd.ExcelFile("crab.xlsx")
dfs = xl_file.parse("Data", header = None) 
dat = np.asarray(dfs)

#create target and feature matrix
y = np.asarray(np.matrix(dat[:,4]).T)
x = dat[:,:4]

#OLS weights
wlr = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
plr = np.matmul(x,wlr)
soslr = sum((y-plr)**2)[0]
mse = ((y-plr)**2).mean(axis=None)

print("OLS LINEAR REGRESSION:")
print("SumOfSquares: " + soslr.astype(str) +
      "\nMeanSquaredErr: " + mse.astype(str))

def poisson(wt):
    wtx = np.matmul(x,wt)
    return(np.exp(wtx))

def gradient(wt):
    p = poisson(wt)
    G = np.matmul(x.T,(p-y))
    P =  np.diag(p.T[0])
    H = np.matmul(np.matmul(x.T,P),x)
    W = np.matmul(np.linalg.inv(H),G)
    return (W)

i = 0
w = wlr
eps=tol
while i < max_iter and eps >= tol:
    w_old = w/sum(abs(w))
    w = w - gradient(w)
    eps = sum(abs(w_old - w / sum(abs(w))))
    i = i+1

print(i,eps)
pglm = poisson(w)
soslr = sum((y-pglm)**2)[0]
mse = ((y-pglm)**2).mean(axis=None)

print("\nGLM (POISSON):")
print("SumOfSquares: " + soslr.astype(str) +
      "\nMeanSquaredErr: " + mse.astype(str))
