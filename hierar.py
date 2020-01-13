#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 04:27:01 2020

@author: pete
"""
import csv
import face_recognition
import os
from PIL import Image
import numpy as np
import sys
 
destdir = '/home/pete/faces8'

files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ] 

all_images=[]
predicted_classes=[]
k=0

#    for file in tests[:33]:
os.chdir(destdir)
k=0 
for file in files:
        k+=1
        print(k)
        known_obama_image = face_recognition.load_image_file(file)
        # Get the face encodings for the known images
        width=known_obama_image.shape[0]
        height=known_obama_image.shape[1]
        face_location=(0,width,height,0)
        face_locations=[face_location]
        obama_face_encoding = face_recognition.face_encodings(known_obama_image,face_locations)[0]
        if k==1:
            arrayf=obama_face_encoding           
        else:
            arrayf=np.vstack([ arrayf,obama_face_encoding])
            
from scipy.cluster import hierarchy
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(arrayf)
names=pd.DataFrame(files)
named=pd.concat([names, df],axis=1,ignore_index=True)
named.to_csv('/home/pete/compare/dataframe1.csv', encoding='utf-8', index=False)
#df = pd.read_csv('embed1.csv', header=None)
#
wine_complete = hierarchy.complete(df)
dn = hierarchy.dendrogram(wine_complete)
#df = df.set_index(names)
#del df.index.name
fig = plt.figure()
#dn = hierarchy.dendrogram(wine_complete,labels=names.names[0])
plt.show()