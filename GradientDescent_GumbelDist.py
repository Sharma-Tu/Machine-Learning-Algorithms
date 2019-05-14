# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 22:03:44 2019
Topic: Script for Gradient Descent Algorithm (Gumbel Dist)
@author: Tushar Sharma
"""

import numpy
import math
import datetime
import matplotlib.pyplot as plt

startnow = datetime.datetime.now()


#Initializing Parameters for gradient descent
eta = numpy.array([0.00001,0.001]);
tau = 0.6;
tol = 0.0001;

"""change data distribution here"""
#Initializing Random Dataset
loc=4.0;scale=5.0;
d = numpy.random.gumbel(loc, scale, size=1000);

#Initialize distribution parameter
w = numpy.array([3.5, 4.0]);

def expo1(w):
    expo = float(0);
    for i in range(len(d)):
        expo = expo + math.exp(-((d[i]-w[0])/w[1]));
    return expo;

def expo2(w):
    expo = float(0);
    for i in range(len(d)):
        expo = expo + (d[i]-w[0])*(math.exp(-((d[i]-w[0])/w[1])));
    return expo;

def grad(w):
    gradloc = ((-len(d)+expo1(w)/w[1]));
    gradscale = (len(d)/w[1]) - ((sum(d)- (len(d)*w[0]))/(w[1]*w[1])) + (expo2(w)/(w[1]*w[1]));
    grad = [gradloc, gradscale];
    return numpy.array(grad);

def func(w):
    return((len(d)*math.log(w[1])) + (sum(d)-(len(d)*w[0]))/w[1] + expo1(w));


for i in range(100):
    obj = func(w);
    temp = (w - (eta * grad(w)));
    llw = func(temp);
    if (llw < numpy.absolute(obj - tol)):
        w=temp;
    else:
        break;

endnow = datetime.datetime.now();

print(loc,scale,w,eta,(endnow-startnow));

#plt.hist(d[d<50]);