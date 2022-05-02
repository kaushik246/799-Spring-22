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
import vive_utils as vu

light_dims = [44.0,42.0,90.0]
light_pts = vu.generate3DBox(light_dims[0],light_dims[1],light_dims[2])

print(light_pts[:,[1]])
print(len(light_pts[0,:]))
