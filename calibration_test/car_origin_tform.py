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

isTesting = 1;

#In image space
#offset_x = -6;
#offset_y = 10;
#y_rot = 0.9*(np.pi/180)
offset_x = 0.0
offset_y = 0.0
y_rot = 0.0
pxls_per_degree_x = 22.1;
pxls_per_degree_y = 18.6;

#in tracker space
x_rot = -(offset_y/pxls_per_degree_y)*(np.pi/180)
z_rot = -(offset_x/pxls_per_degree_x)*(np.pi/180)

filename = sys.argv[1]
print("using: " + filename);

mat_bottom = np.array([[0.0,0.0,0.0,1.0]]);
origin = np.array([[0.0],[0.0],[0.0],[1.0]]);

f = open(filename);
reader = csv.reader(f);

row = next(reader);
#tform to vehicle
To_v = [[float(row[0]),float(row[1]),float(row[2]),float(row[3])],[float(row[4]),float(row[5]),float(row[6]),float(row[7])],[float(row[8]), float(row[9]),float(row[10]),float(row[11])]]

To_v = np.append(To_v,mat_bottom,axis=0);
row = next(reader);
#tform to camera
To_c = [[float(row[0]),float(row[1]),float(row[2]),float(row[3])],[float(row[4]),float(row[5]),float(row[6]),float(row[7])],[float(row[8]), float(row[9]),float(row[10]),float(row[11])]]
To_c = np.append(To_c,mat_bottom,axis=0);

print(To_v)
print(To_c)

Tv_o = np.linalg.inv(To_v);
Tv_c = np.matmul(Tv_o,To_c);

dist = np.matmul(Tv_c,origin);
print(dist)
dist[0][0] = dist[0][0] + .0350;
dist[2][0] = dist[2][0] - .0315;

x_rot_mat = np.array([[1,0,0],[0,np.cos(x_rot),-np.sin(x_rot)],[0,np.sin(x_rot),np.cos(x_rot)]]);
y_rot_mat = np.array([[np.cos(y_rot),0,np.sin(y_rot)],[0,1,0],[-np.sin(y_rot),0,np.cos(y_rot)]]);
z_rot_mat = np.array([[np.cos(z_rot),-np.sin(z_rot),0],[np.sin(z_rot),np.cos(z_rot),0],[0,0,1]]);

total_rot = np.matmul(x_rot_mat,z_rot_mat);
total_rot = np.matmul(y_rot_mat,total_rot);

final_v_to_c = np.append(total_rot,np.array([[0.0,0.0,0.0]]),axis=0); 
final_v_to_c = np.append(final_v_to_c,dist,axis=1)

#TEST:
if (isTesting):
	dist[0][0] = 0.0;
	dist[1][0] = .2;
	dist[2][0] = 0.0;
	final_v_to_c = np.array([[1.0,0.0,0.0,dist[0][0]],[0.0,1.0,0.0,dist[1][0]],[0.0,0.0,1.0,dist[2][0]],[0.0,0.0,0.0,1.0]])
print(final_v_to_c)

pickle.dump(final_v_to_c,open("vehicle_to_camera_offset.p","wb"))

