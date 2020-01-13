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

with open('embed2.csv','w',newline='') as csvfile:
#    for file in tests[:33]:
    os.chdir(destdir)
    writer=csv.writer(csvfile,quoting=csv.QUOTE_NONE)
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
        str1=np.array2string(obama_face_encoding,separator=',')
        str2=str1[1:-1]
 #       writer.writerow([(file),str2])
        
        writer.writerow([(file),obama_face_encoding])
