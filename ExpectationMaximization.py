# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:07:27 2019
Topic: Script for EM Algorithm (Mixture of 2 Gaussian)
@author: Tushar
"""

import numpy
import math
import matplotlib.pyplot as plt

#Generate Random Data for 2 Ns
w = numpy.array([0.4,0.6]); #true 
n = 10000;
mu = numpy.array([30.0,6.0]);
sigma = numpy.array([1.0,1.5]);
data =numpy.empty(n);
d=0.0;

def randnmix():
    a = numpy.random.binomial(1,w[1]);
    d1 = numpy.random.normal(mu[a],sigma[a],1)
    #d2 = ((1/math.sqrt(2*math.pi*(sigma[1]*sigma[1])))*math.exp(-(x-mu[1])/(sigma[1]*sigma[1])));
    #d = d1+d2;
    return d1;

for i in range(n):
    data[i] = randnmix();

def normal(x,a,mu,sigma):
    return((1/numpy.sqrt(2*math.pi*(sigma[a]**2)))*numpy.exp(-((x-mu[a])**2)/(2*(sigma[a]**2))));

def pycalc(x,w,mu,sigma):
    py1 = ((w[0]*normal(x,0,mu,sigma))/((w[0]*normal(x,0,mu,sigma))+((1-w[0])*normal(x,1,mu,sigma))));
    py2 = (1-py1);
    return(numpy.array([py1,py2]));

py = numpy.empty([n,2]);
muit = numpy.array([25.0,5.0]);
sigmait = numpy.array([2.5,2.5]);
wit = numpy.array([0.3,0.7]);


for k in range(40):
    muitint = numpy.array([0,0]);
    sigmaitint = numpy.array([0,0]);
    for q in range(n):
        py[q] = pycalc(data[q],wit,muit,sigmait);
        #print(py[q],data[q],wit,muit,sigmait);
        #if (q==503):
            #print(data[q],wit,muit,sigmait,py[q]);
    beta = sum(py[:,0])/n;
    wit =  numpy.array([beta,1-beta]);
    for z in range(n):
        muitint = muitint + (py[z]*data[z]);
    muit = numpy.array([(muitint[0]/sum(py[:,0])),(muitint[1]/sum(py[:,1]))]);
    for l in range(n):
        sigmaitint = sigmaitint + (py[l]*((numpy.array(data[l],data[l])-muit)**2));
    sigmait = numpy.array([(sigmaitint[0]/sum(py[:,0])),(sigmaitint[1]/sum(py[:,1]))]);
    
print(w,mu,sigma, wit,muit,sigmait);

#print(data);
#pycalc(data[1],numpy.array([0.3,0.7]),numpy.array([0.9,7.5]),numpy.array([0.6,1.7]));

#plt.hist(data[data<50]);