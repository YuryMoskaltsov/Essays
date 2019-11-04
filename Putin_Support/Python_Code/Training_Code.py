#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 23:51:27 2019

@author: yurymoskaltsov
"""

from matplotlib import pylab as plt
import seaborn as sns
import numpy as np

x = [1,2,3,4,5,6]
y = [-i**2 for i in x]
z = [i*2 for i in x]
m = [17,21,4,89,35,108]
print(m)

#sns.distplot(x, bins = 20)
#sns.boxplot(x)
#sns.lineplot(x,y)

plt.figure()
plt.subplot(121)
plt.plot(x,z)
plt.subplot(122)
plt.plot(x,y)
#plt.show()

#print(np.corrcoef(x,m))