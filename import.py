#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 04:27:01 2020

@author: pete
"""

import face_recognition
import os
from PIL import Image
import numpy as np
import sys
 
destdir = '/home/pete/faces8'

files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ] 

all_images=[]
predicted_classes=[]

#'''

known_obama_image = face_recognition.load_image_file("/home/pete/faces8/m1.jpg")
known_biden_image = face_recognition.load_image_file("/home/pete/faces8/m3.jpg")

# Get the face encodings for the known images
width=known_obama_image.shape[0]
height=known_obama_image.shape[1]

face_location=(0,width,height,0)
face_locations=[face_location]
obama_face_encoding = face_recognition.face_encodings(known_obama_image,face_locations)[0]

width=known_biden_image.shape[0]
height=known_biden_image.shape[1]

face_location=(0,width,height,0)
face_locations=[face_location]
biden_face_encoding = face_recognition.face_encodings(known_biden_image,face_locations)[0]
known_face_encodings = [
    obama_face_encoding
#    biden_face_encoding
]

face_distances = face_recognition.face_distance(known_face_encodings, biden_face_encoding)
sys.exit(0)
# Load a test image and get encondings for it
image_to_test = face_recognition.load_image_file("/home/pete/faces8/obama2.jpg")


image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]
