#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:51:14 2019

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

# See how far apart the test image is from the known faces
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    print()

#'''

locat_all=[]

for f in files:
      print(f)     
      image = face_recognition.load_image_file(os.path.join(destdir,f))
      width, height = image.size     
#      face_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
      obama_face_encoding = face_recognition.face_encodings(known_obama_image)[0]
      
      face_locations = face_recognition.face_locations(image)
      face_landmarks_list = face_recognition.face_landmarks(image)
      all_images.append(face_landmarks_list)
      locat_all.append(face_locations)
      for face_location in face_locations:
          top, right, bottom, left = face_location
          face_image=image[top:bottom, left:right]
          pil_image = Image.fromarray(face_image)
#          pil_image.show()         
#          predicted_class = np.argmax(model.predict(face_image))
#         predicted_classes.append(predicted_class)
          