# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 01:14:37 2019
Topic: Script for Gradient Descent Algorithm (Mixture of 2 Gaussian)
@author: Tushar
"""


import numpy
import math
import matplotlib.pyplot as plt

#Generate Random Data for 2 Ns
#w= w1,w2,mu1,mu2,sigma1,sigma2
wtrue = numpy.array([0.4,0.6,30.0,6.0,1.0,1.5]); #true 
n = 1000;
data =numpy.empty(n);
d=0.0;
eta = numpy.array([0.0001,0.0001,0.0001,0.000007,0.0001,0.0000001]);
tol=0.0001;

def randnmix():
    a = numpy.random.binomial(1,wtrue[1])+2;
    b = a+2;
    d1 = numpy.random.normal(wtrue[a],wtrue[b],1)
    return d1;


for i in range(n):
    data[i] = randnmix();


#Initialize parameters
w = numpy.array([0.3,0.7,29.0,4.0,0.8,1.0]);

def normal(x,mu,sigma):
    return((1/numpy.sqrt(2*math.pi*(sigma**2)))*numpy.exp(-((x-mu)**2)/(2*(sigma**2))));


def pycalc(w):
    py1 = 0;py3 = 0;py4=0;py5=0;py6=0;
    temppy=0;
    for i in range(len(data)):
        temppy = ((w[0]*normal(data[i],w[2],w[4]))/((w[0]*normal(data[i],w[2],w[4]))+((1-w[0])*normal(data[i],w[3],w[5]))));
        py1 = py1 + temppy;
        py3 = py3 + temppy*((data[i]-w[2])/(w[4]**2));
        py4 = py4 + temppy*((data[i]-w[3])/(w[5]**2));
        py5 = py5 + temppy*((-1/w[4])+(((data[i]-w[2])**2)/(w[4]**3)));
        py6 = py6 + temppy*((-1/w[5])+(((data[i]-w[3])**2)/(w[5]**3)));
    #py2 = (1-py1);
    return(numpy.array([py1,py3,py4,py5,py6]));


def grad(w):
    py = pycalc(w);
    gradw1 = (py[0]/w[0])-len(data);
    gradmu1 = py[1];
    gradmu2 = py[2];
    gradsigma1 = py[3];
    gradsigma2 = py[4];
    grad = [-gradw1,-gradw1, -gradmu1, -gradmu2, -gradsigma1, -gradsigma2];
    return numpy.array(grad);

def objective(w):
    temppy1=0;temppy2=0;obj=0;
    for j in range(len(data)):
        temppy1 = ((w[0]*normal(data[j],w[2],w[4]))/((w[0]*normal(data[j],w[2],w[4]))+((1-w[0])*normal(data[j],w[3],w[5]))));
        temppy2 = ((w[1]*normal(data[j],w[3],w[5]))/((w[1]*normal(data[j],w[3],w[5]))+((1-w[1])*normal(data[j],w[2],w[4]))));
        obj = obj + (numpy.log(w[0]*normal(data[j],w[2],w[4]))*temppy1)+(numpy.log(w[1]*normal(data[j],w[3],w[5]))*temppy2);
    return(obj-(len(data)*(w[0]+w[1]))-1);

for k in range(50):
    obj = -objective(w);
    temp = (w - (eta * grad(w)));
    temp[1]=1-temp[0];
    llw = -objective(w);
    if (llw < numpy.absolute(obj - tol)):
        break;
    else:
        w=temp;

#endnow = datetime.datetime.now();

print(wtrue, w);

#plt.hist(data[data<50]);