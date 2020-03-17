import dlib
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
def Distance(a,b):  
     dist = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)  
     return dist  

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/path/shape_predictor_68_face_landmarks.dat")

from google.colab.patches import cv2_imshow
image = cv2.imread("/content/unnamed.jpg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)
cv2_imshow(gray)

import pandas as pd
for (i, rect) in enumerate(rects):
  shape = predictor(gray, rect)
  shape = face_utils.shape_to_np(shape)
  dis=Distance(shape[0],shape[16])
  print(dis)
