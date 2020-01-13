#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 05:34:06 2020

@author: pete
"""
from scipy.cluster import hierarchy
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('embed1.csv', sep=',', header=None)
#df = pd.read_csv('embed1.csv', header=None)
wine_complete = hierarchy.complete(df)
fig = plt.figure()
dn = hierarchy.dendrogram(wine_complete)
plt.show()