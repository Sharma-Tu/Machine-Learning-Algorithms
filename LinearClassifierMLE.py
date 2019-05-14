# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:46:48 2019

@author: Tushar
"""

import numpy as np
import matplotlib.pyplot as plt
#import time

from sklearn import linear_model

#INITIALIZE
npos = 200
nneg = 200
mu1 = np.array([1,1])
mu2 = np.array([10,10])
sig1 = np.array([[1,0],[0,1]])
sig2 = np.array([[1,0],[0,1]])
max_step = 2e5
eta = 1e-4
tol = 1e-6

print(eta,tol)
#randomly generated data for classifier
pos = np.random.multivariate_normal(mean=mu1, cov=sig1, size=npos)
neg = np.random.multivariate_normal(mean=mu2, cov=sig2, size=nneg)
ones = np.ones((npos+nneg, 1))
dat = np.r_[pos,neg]
dat = np.c_[ones, dat]

#def logsig(wg):
#    lg = 1/(1+np.exp(-wg.T*x))
#    return lg

#write gradient and hessian function
def grad(wg):
    wtxsq = np.square(np.matmul(x,wg))
    sqrtwtxsq = np.sqrt(1+wtxsq)
    g = (x*(2*y - 1)*sqrtwtxsq-np.matmul(x,wg))/(1+wtxsq)
    g = np.matrix(np.sum(g, axis=0)).T
    return(np.asarray(g))

#def gradlg(wg):
#    p = logsig(wg)
#    P = np.diag(np.multiply(p,(1 - p)))
#    #E = np.diag(y-p)
#    #J = P*x
#    #X' * P * (y - p)
#    g = x * P * (y-p)
#    return (g)

x = dat
y = np.ones((npos,1))
y = np.r_[y,(np.zeros((nneg,1)))]

#initial solution using OLS
wi = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)

# PLOT function
def xyplot(beta):
    mn = - beta[0] / beta[1] - beta[2] / beta[1] * min(x[:, 2])
    mx = - beta[0] / beta[1] - beta[2] / beta[1] * max(x[:, 2])
    x1 = np.linspace(mn,mx,10);
    x2 = - beta[0] / beta[2] - beta[1] / beta[2] * x1
    colors = np.squeeze(np.asarray(y))
    plt.scatter(dat[:,1],dat[:,2], c = colors, alpha=0.7)
    plt.plot(x1,x2, "r--")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def xyplot1(beta1,beta2):
    mn1 = - beta1[0] / beta1[1] - beta1[2] / beta1[1] * min(x[:, 2])
    mx2 = - beta1[0] / beta1[1] - beta1[2] / beta1[1] * max(x[:, 2])
    x11 = np.linspace(mn1,mx2,20);
    x21 = - beta1[0] / beta1[2] - beta1[1] / beta1[2] * x11
    mn3 = - beta2[0] / beta2[1] - beta2[2] / beta2[1] * min(x[:, 2])
    mx4 = - beta2[0] / beta2[1] - beta2[2] / beta2[1] * max(x[:, 2])
    x12 = np.linspace(mn3,mx4,20);
    x22 = - beta2[0] / beta2[2] - beta2[1] / beta2[2] * x12
    #mn5 = - beta3[0] / beta3[1] - beta3[2] / beta3[1] * min(x[:, 2])
    #mx6 = - beta3[0] / beta3[1] - beta3[2] / beta3[1] * max(x[:, 2])
    #x13 = np.linspace(mn5,mx6,20);
    #x23 = - beta3[0] / beta3[2] - beta3[1] / beta3[2] * x13
    colors = np.squeeze(np.asarray(y))
    plt.scatter(dat[:,1],dat[:,2], c = colors, alpha=0.7)
    plt.plot(x11,x21, "r-", color = "green", linewidth = 0.7)
    plt.plot(x12,x22, "r--", label = "", linewidth = 0.7)
    plt.legend(["Given Function","Logistic Function"])
    #plt.plot(x13,x23, "r--", color = "blue", linewidth=0.5)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

xyplot(wi)
print(wi)

#iteratively solve for optimum W parameters
i=0
w=wi
eps = tol
while i < max_step and eps >= tol:
    w_old = w/sum(abs(w))
    #w_old = w/sum(w)
    w = w+(eta*grad(w))
    eps = sum(abs(w_old - w / sum(abs(w))))
    #eps = sum(w_old - w / sum(w))
    #print(eps)
    #xyplot(w)
    i = i+1

xyplot(w)
print(i,eps)

print(w)

clf = linear_model.SGDClassifier(loss = 'log',
                                 max_iter = 1e5,
                                 learning_rate = 'constant',
                                 eta0 = 1e-4,
                                 tol = 1e-6,
                                 fit_intercept = False,
                                 shuffle=False).fit(x,np.squeeze(y))

xyplot(clf.coef_.T)
print(clf.n_iter_)
print(clf.coef_.T)

xyplot1(w,clf.coef_.T)
