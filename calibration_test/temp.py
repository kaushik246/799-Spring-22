#! /usr/bin/python3
import os
import numpy as np
import cv2
import PIL
#from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys
import glob
import csv
from scipy.spatial.transform import Rotation as R


cam_matrix = pickle.load(open("calibration/cam_matrix.p","rb"))
print(cam_matrix)

dist_matrix = pickle.load(open("calibration/dist_matrix.p","rb"))
print(dist_matrix)

new_cam_matrix = [[1374.40405273438,0.0,978.021118164062],[0.0,1375.505371109375,553.55859375],[0.0,0.0,1.0]]
new_dist_matrix = [[0.0,0.0,0.0,0.0,0.0]]	
pickle.dump(new_cam_matrix,open("cam_matrix_intel.p","wb"))
pickle.dump(new_dist_matrix,open("dist_matrix_intel.p","wb"))

