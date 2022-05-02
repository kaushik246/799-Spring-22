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
import math
import vive_utils as vu
from scipy.spatial.transform import Rotation as R
plt.figure()

#offset_x = -6;
#offset_y = 13;

offset_x = 0;
offset_y = 0;

#just the light:
#light_dims = [44.0*0.001,42.0*0.001,59.0*0.001]
light_dims = [44.0*0.001,42.0*0.001,89.0*0.001]
light_pts = vu.generate3DBox(light_dims[0],light_dims[1],light_dims[2])

cam_matrix = pickle.load(open("calibration/cam_matrix_intel.p","rb"))
dist_matrix = pickle.load(open("calibration/dist_matrix_intel.p","rb"))
car_to_camera = pickle.load(open("vehicle_to_camera_offset.p","rb"))
car_to_camera = np.array(car_to_camera);
origin = np.array([[0.0],[0.0],[0.0],[1.0]]);
identity = np.float64([[1,0,0],[0,1,0],[0,0,1]]);
mat_bottom = np.array([[0.0,0.0,0.0,1.0]]);
twizzle_mat = [[-1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0]];

def projectPointAtOrigin(To_t, To_p):
		To_t = np.matmul(To_t,car_to_camera); # Vive origin
		Tt_o =	np.linalg.inv(To_t); # Transform
		Tt_p = np.matmul(Tt_o,To_p); # Transform between camera and target (light)
		Tt_o = np.matmul(twizzle_mat,Tt_o);
		Tt_p = np.matmul(twizzle_mat,Tt_p); 
		localCoords = np.matmul(Tt_p,origin)
		twizzleCoord = [[localCoords[0][0]],[localCoords[1][0]],[localCoords[2][0]]]
		imagePts,jacobian = cv2.projectPoints(np.float64(twizzleCoord),identity,np.float64([[0],[0],[0]]),np.float64(cam_matrix),np.float64(dist_matrix))
		return imagePts;

def projectPoint(To_t, To_p, pts):
		#pts = np.matmul(To_p,pts);
		To_t = np.matmul(To_t,car_to_camera);
		Tt_o = np.linalg.inv(To_t);
		Tt_p = np.matmul(Tt_o,To_p);
		pts = np.matmul(Tt_p,pts);
		pts = np.matmul(twizzle_mat,pts);
		twizzleCoord = [[pts[0][0]],[pts[1][0]],[pts[2][0]]]
		imagePts,jacobian = cv2.projectPoints(np.float64(twizzleCoord),identity,np.float64([[0],[0],[0]]),np.float64(cam_matrix),np.float64(dist_matrix))
		return imagePts;

def getDistAndAngles(To_t,To_p):
		To_t = np.matmul(To_t,car_to_camera);
		Tt_o =	np.linalg.inv(To_t);
		Tt_p = np.matmul(Tt_o,To_p);
		localCoords = np.matmul(Tt_p,origin)
		distance = math.sqrt(localCoords[0][0]**2 +localCoords[1][0]**2 +localCoords[2][0]**2)
		angle = math.atan2(localCoords[0][0],localCoords[1][0])
		angle = angle*(180/np.pi)
		elevationAngle = math.atan2(-localCoords[2][0],localCoords[1][0])
		elevationAngle = elevationAngle*(180/np.pi)
		return distance,angle,elevationAngle

def getLocalCoords(To_t,To_p):
		To_t = np.matmul(To_t,car_to_camera);
		Tt_o =	np.linalg.inv(To_t);
		Tt_p = np.matmul(Tt_o,To_p);
		localCoords = np.matmul(Tt_p,origin)
		return localCoords 

def getBoundingBox(ptsIn):
		x = []
		y = []
		for elem in ptsIn:
			item1, item2 = elem
			x.append(item1)
			y.append(item2)
		
		x_min = min(x)
		y_min = min(y)
		x_max = max(x)
		y_max = max(y)
		return x_min, y_min, x_max, y_max


def rowToMatrix(csvRow):
		newArray = np.array(csvRow);
		toReturn = np.reshape(newArray,(3,4));
		toReturn = np.append(toReturn,mat_bottom,axis=0);
		return toReturn;

def getIOU(boxA, boxB):
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
		# compute the area of intersection rectangle
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
		# return the intersection over union value
		return iou

def generate_dict(file1):
		toReturn = {} 
		with open(file1, newline='') as text1:
			file1reader = csv.reader(text1, delimiter=',')

			for row in file1reader:
				img_name = row[0]
				x_min = float(row[1])
				y_min = float(row[2])
				x_max = float(row[3])
				y_max = float(row[4])

				box1 = [x_min, y_min, x_max, y_max]

				if img_name not in toReturn:
					toReturn[img_name] = []

				toReturn[img_name] = box1
			return toReturn

if (len(sys.argv) > 2):
	isPrinting = int(sys.argv[2])
else:
	isPrinting = 1;

folderPath = sys.argv[1];
labels = generate_dict(folderPath + "/labels.txt")
images = sorted(glob.glob(folderPath + '/*.png'),key=os.path.getmtime)

light_location = np.array([[0.0,0.0,0.0]]);
#transform between tracker and camera
track_cam = [[0,0,0],[0,0,0],[0,0,0]];

#lightFile = open(folderPath + '/light_location_nocar.csv',mode='r');
#lightReader = csv.reader(lightFile);
#lightLocations = [[float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11])] for row in lightReader]

car_locations = {}

csvFile = open(folderPath + '/locations.txt',mode = 'r');
reader = csv.reader(csvFile);
car_locations = {}
light_locations = {}
for row in reader:
		car_locations[row[0]] = [[float(row[1]),float(row[2]),float(row[3]),float(row[4])],[float(row[5]),float(row[6]),float(row[7]),float(row[8])],[float(row[9]), float(row[10]),float(row[11]),float(row[12])]]
		light_locations[row[0]] = [[float(row[13]),float(row[14]),float(row[15]),float(row[16])],[float(row[17]),float(row[18]),float(row[19]),float(row[20])],[float(row[21]), float(row[22]),float(row[23]),float(row[24])]]
for image in images:
	try:
		basename = os.path.basename(image);
		labeledBox = labels[basename]
		to_t = np.array(car_locations[basename]);
		to_t = np.append(to_t,mat_bottom,axis=0);
		to_l = np.array(light_locations[basename]);
		to_l = np.append(to_l,mat_bottom,axis=0);
		imagePt = projectPointAtOrigin(to_t,to_l);
		imagePt = (np.int32(imagePt[0][0][0])+offset_x,np.int32(imagePt[0][0][1]+offset_y))
		pt2 = projectPoint(to_t,to_l,origin)
		boundPts = []
		for i in range(0,len(light_pts[0,:])):
			newPoint =projectPoint(to_t,to_l,light_pts[:,[i]])
			newPoint = (np.int32(newPoint[0][0][0]+offset_x),np.int32(newPoint[0][0][1]+offset_y))
			boundPts.append(newPoint)
		bx1, by1, bx2, by2 = getBoundingBox(boundPts)
		iou = getIOU([bx1,by1,bx2,by2],labeledBox )
		distance,angle,elevation = getDistAndAngles(to_t,to_l);
		localCoords = getLocalCoords(to_t,to_l);
		
		#coordinates here are y is straight, x is to the right
		calc_x = (bx1+bx2)/2;
		calc_y = (by1+by2)/2;
		label_x = (labeledBox[0] + labeledBox[2])/2
		label_y = (labeledBox[1] + labeledBox[3])/2
		outputString = "OUTPUT,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (basename,-localCoords[0][0],localCoords[1][0],iou,distance,angle,elevation,label_x,label_y,calc_x,calc_y)
		print(outputString)
		if (isPrinting):
			img = cv2.imread(image);
			img = cv2.putText(img,outputString, (25,25),cv2.FONT_HERSHEY_SIMPLEX,1,color=(255,255,255))
			img = cv2.rectangle(img,(np.int32(labeledBox[0]),np.int32(labeledBox[1])),(np.int32(labeledBox[2]),np.int32(labeledBox[3])),color=(0,255,0),thickness=2);
			img = cv2.rectangle(img,(bx1,by1),(bx2,by2),color=(0,128,255),thickness=2);
			pt2 = (np.int32(pt2[0][0][0]),np.int32(pt2[0][0][1]))
			img = cv2.circle(img,pt2,radius=5,color=(255,255,255),thickness=-1);
			img = cv2.circle(img,imagePt,radius=5,color=(0,0,255),thickness=-1);
			img = cv2.circle(img,imagePt,radius=5,color=(0,0,255),thickness=-1);
			img = cv2.line(img,(960,0),(960,1080),color=(255,0,0),thickness=1);
			img = cv2.line(img,(0,560),(1920,560),color=(255,0,0),thickness=1);
			cv2.imshow('window',img)
			cv2.waitKey(0)
	except KeyError as ke:
		#print(ke)
		continue
