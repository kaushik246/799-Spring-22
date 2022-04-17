import cv2
import numpy as np
import os
import glob
import imutils
import sys
import math


class Camera:
    def __init__(self, path, dim):
        self.path = path
        self.cam_mat = []
        self.checkerboard_dim = dim
        self.obj_points = []
        self.img_points = []

    def calibrate(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        obj_p = np.zeros((1, self.checkerboard_dim[0] * self.checkerboard_dim[1], 3), np.float32)
        obj_p[0,:,:2] = np.mgrid[0:self.checkerboard_dim[0], 0:self.checkerboard_dim[1]].T.reshape(-1, 2)

        images = glob.glob(self.path + '/*.jpg')

        for filename in images:
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dim,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret == True:
                self.obj_points.append(obj_p)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.img_points.append(corners2)

            h, w = img.shape[:2]
            ret, self.cam_mat, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, gray.shape[::-1], None, None)
            '''
            print("Camera matrix : \n")
            print(self.cam_mat)
            print("dist : \n")
            print(dist)
            print("rvecs : \n")
            print(rvecs)
            print("tvecs : \n")
            print(tvecs)
            '''

class TagDetector:
    def __init__(self, img_path, marker_type):
        self.img_path = img_path
        self.marker_type = marker_type
        self.aruco_dict = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        }
        self.points ={
            'x': [],
            'y': []
        }

    def detect_markers(self):
        image = cv2.imread(self.img_path)
        image = imutils.resize(image, width=600)

        if self.aruco_dict.get(self.marker_type, None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(
                self.marker_type))
            sys.exit(0)

        arucoDict = cv2.aruco.Dictionary_get(self.aruco_dict[self.marker_type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

        if len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                self.points['x'].append(cX)
                self.points['y'].append(cY)

    def compute_center(self):
        n = len(self.points['x'])
        c_x = sum(self.points['x'])/n
        c_y = sum(self.points['y'])/n
        import pdb
        pdb.set_trace()
        return (c_x, c_y)




class Calibration:
    def __init__(self, img_path, marker_type, cam_dict={}):
        self.img_path = img_path
        self.marker_type = marker_type
        if not cam_dict:
            cam = Camera(cam_dict.get('img'), cam_dict.get('dim'))
            cam.calibrate()
            self.cam_mat = cam.cam_mat
        else:
            self.cam_mat = cam_dict.get('cam_mat')

        self.detector = TagDetector(self.img_path, self.marker_type)


    @staticmethod
    def get_distance(point_1, point_2):
        return math.sqrt((point_1[0]-point_2[0])**2 + (point_1[0]-point_2[0])**2)

    def get_3d_points(self):
        Identity = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])



if __name__ == '__main__':
    checkerboard_path = "./images"
    camera_obj = Camera(checkerboard_path, (6, 9))
    camera_obj.calibrate()

    tag_path = "./tags/tags"
    tag_detection_obj = TagDetector(tag_path, "DICT_6X6_250")
    tag_detection_obj.detect_markers()
    (c_x, c_y) = tag_detection_obj.compute_center()

    for i in range(1, 50):
        tag_path += '_' + str(i) + '.png'
        tag_detection_obj = TagDetector(tag_path, "DICT_6X6_250")
        tag_detection_obj.detect_markers()
        (c_x, c_y) = tag_detection_obj.compute_center()

        tag_path = "./tags/tags"