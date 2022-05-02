#! /usr/bin/python3
import numpy as np
import cv2
import glob
import pickle 
import sys

filePath = sys.argv[1]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, .001)

objp = np.zeros((4*11,3), np.float32)

objp[:,:2] = np.mgrid[0:4,0:11].T.reshape(-1,2)
print(objp.shape)

objp=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[0.5,0.5,0],[1.5,0.5,0],[2.5,0.5,0],[3.5,0.5,0]],np.float32)
for y in range(2,11):
            for x in range(4):
                                objp=np.append(objp,[np.array([objp[4*(y-2)+x][0],objp[4*(y-2)+x][1]+1,0],np.float32)],axis=0)
print(objp.shape)
print(objp)

objpoints = []
imgpoints = []

images = glob.glob(filePath + '/*.png')

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 10
params.maxArea = 10000
blobDetector = cv2.SimpleBlobDetector_create(params)
for fname in images:
    img=cv2.imread(fname);
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # ret, corners = cv2.findCirclesGrid(gray,(11,4),flags = cv2.CALIB_CB_ASYMMETRIC_GRID)

    ret, corners = cv2.findCirclesGrid(gray,(4,11),flags=cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING)
    print(fname + ":" )
    if (corners is not None):
        for i in range(0,len(corners)):
            img = cv2.circle(img, (corners[i][0][0],corners[i][0][1]),5,(255,0,0),3)
    if ( ret == True):
        objpoints.append(objp)
        #print(len(corners))
        #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        print("circles found")
        corners2 = corners
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img,(4,11),corners2,ret)
#        cv2.imshow('img',img)
#        cv2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
print("CALIBRATION ERROR (should be less than .5px for good results): %f" % ret)
pickle.dump(mtx,open("cam_matrix.p","wb"))
pickle.dump(dist,open("dist_matrix.p","wb"))
print(mtx)
print(dist)
cv2.destroyAllWindows()
