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
import requests

'''
AXIS DEFINTIONS:
Tracker_forward (from mounting screw to LED along the bottom) : y_hat_t
Tracker_left (same plane as bottom, orthogonol to y_hat and left) : x_hat_t
Tracker_down (orthogonal to the other 2, right-handed) : z_hat_t

Camera_forward (orthogonal to camera plane, leaving from principal point): z_hat
Camera_down (straight down): y_hat_c
Camera_right (going along right side of image): x_hat_c

To convert between them:
x_hat_c = -x_hat_t
y_hat_c = z_hat_t
z_hat_c = z_hat_t
'''

mat_bottom = np.array([[0.0,0.0,0.0,1.0]]);
origin_const = np.array([[0.0],[0.0],[0.0],[1.0]]);

def lineTo4x3(line):
	values = line.split(',');
	if (len(values) > 12):
		values = values[0:12]
	toReturn = []
	for value in values:
		toReturn.append(float(value));
	toReturn = np.array(toReturn)
	toReturn = np.reshape(toReturn,(3,4));
	toReturn = np.append(toReturn,mat_bottom,axis=0);
	return toReturn;

def lineTo4x4(line):
	values = line.split(',');
	if (len(values) > 16):
		values = values[0:16]
	toReturn = []
	for value in values:
		toReturn.append(float(value));
	toReturn = np.array(toReturn)
	toReturn = np.reshape(toReturn,(4,4));
	return toReturn;

def getDirectMatrix(origin,originOffset,remote,remoteOffset):
	newOrigin = np.matmul(origin,originOffset);
	newOrigin =  np.linalg.inv(newOrigin);
	newRemote = np.matmul(remote,remoteOffset);	
	toReturn = np.matmul(newOrigin,newRemote);
	return toReturn

def getTranslationFrom4x4(inputMatrix):
	return np.matmul(inputMatrix,origin_const)

def flatten4x4(inputMatrix):
	toReturn = ""
	for row in inputMatrix:
		for val in row:
			toReturn = toReturn + "%0.8f," % val
	return toReturn

def getTrackerLocationWeb(trackerName):
	location = requests.get('http://10.42.0.1:4444/track/' + trackerName)
	return location.text


# makes a 4x8 numpy array from dimensions for a 3d bounding box.
# x_hat and y_hat are assumed in the middle of the object, z is at the top, pointed down
# In other words, the tracker is sitting on top.
def generate3DBox(x_dim, y_dim, z_dim):
    toReturn = np.empty((4,0),dtype=float)
    for x in range(0,2):
        x_mul = x-0.5;
        for y in range(0,2):
            y_mul = y-0.5
            for z in range(0,2):
                    z_mul = z;
                    toReturn = np.append(toReturn,[[x_mul*x_dim],[y_mul*y_dim],[z_mul*z_dim],[1.0]],axis=1);
    return toReturn
