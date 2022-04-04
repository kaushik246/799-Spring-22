#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob


# Checkerboard Dimensions
checkerboard_dim = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Array to store 3D points of each checkerboard image
obj_points = []
# Array to store 2D points of each checkerboard image
img_points = []

# Define world co-ordinates
obj_p = np.zeros((1, checkerboard_dim[0] * checkerboard_dim[1], 3), np.float32)
obj_p[0,:,:2] = np.mgrid[0:checkerboard_dim[0], 0:checkerboard_dim[1]].T.reshape(-1, 2)
prev_img_shape = None

images = glob.glob('./images/checkerboard.jpg')
for filename in images:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        obj_points.append(obj_p)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display corners
        img = cv2.drawChessboardCorners(img, checkerboard_dim, corners2, ret)

    cv2.imshow('img', img)
    cv2.waitKey(0)

h,w = img.shape[:2]

# Making camera calibration by passing 3d points and correponding 2d co-ordinates
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)






