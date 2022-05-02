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
import pickle
#import vive_utils as vu

from scipy.spatial.transform import Rotation as R

from detection import TagDetector

class Projection:
    def __init__(self, dir_path):
        self.car_locations = {}
        self.target_locations = {}
        self.offset = [0, 0]
        self.dir_path = dir_path
        self.light_dims = [44.0 * 0.001, 42.0 * 0.001, 89.0 * 0.001]
        self.light_pts = vu.generate3DBox(self.light_dims[0], self.light_dims[1], self.light_dims[2])
        self.twizzle_mat = [[-1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0]]
        self.cam_matrix = pickle.load(open("cam_matrix_intel.p", "rb"))
        self.dist_matrix = pickle.load(open("dist_matrix_intel.p", "rb"))
        self.car_camera_matrix = pickle.load(open("car_camera_matrix.p", "rb"))
        self.origin = np.array([[0.0], [0.0], [0.0], [1.0]]);

    def project_points(self, t_car, t_target):
        t_camera = np.matmul(t_car, self.car_camera_matrix)
        t_camera_inv = np.linalg.inv(t_camera)
        t_camera_target = np.matmul(t_camera_inv, t_target)

        t_camera_inv = np.matmul(self.twizzle_mat, t_camera_inv)
        t_camera_target = np.matmul(self.twizzle_mat, t_camera_target)

        local_coords = np.matmul(t_camera_target, self.origin)
        twizzle_coord = [[local_coords[0][0]], [local_coords[1][0]], [local_coords[2][0]]]
        image_pts, jacobian = cv2.projectPoints(np.float64(twizzle_coord), self.identity, np.float64([[0], [0], [0]]),
                                               np.float64(self.cam_matrix), np.float64(self.dist_matrix))
        return image_pts

    def parse_location_data(self):

        csvFile = open(self.dir_path + '/locations.txt', mode='r')
        reader = csv.reader(csvFile)
        images = sorted(glob.glob(self.dir_path + '/*.png'), key=os.path.getmtime)

        target_location = np.array([[0.0, 0.0, 0.0]]);
        track_cam = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

        for row in reader:
            self.car_locations[row[0]] = [[float(row[1]), float(row[2]), float(row[3]), float(row[4])],
                                     [float(row[5]), float(row[6]), float(row[7]), float(row[8])],
                                     [float(row[9]), float(row[10]), float(row[11]), float(row[12])]]

            self.target_locations[row[0]] = [[float(row[13]), float(row[14]), float(row[15]), float(row[16])],
                                    [float(row[17]), float(row[18]), float(row[19]), float(row[20])],
                                    [float(row[21]), float(row[22]), float(row[23]), float(row[24])]]

        for image in images:
            image_index = os.path.basename(image).split('_')[0][5:]

            to_c = self.car_locations[image_index]

            tag_detection_obj = TagDetector(image, "DICT_6X6_250")
            tag_detection_obj.detect_markers()
            (c_x, c_y, tags_detected) = tag_detection_obj.compute_center()

            if tags_detected == 4:
                print("Aruco Tag Computed Center: " + image + ": " + str(c_x) + ',' + str(c_y))
                print(image_index)
            else:
                print("Image Index: " + image_index + "has " + str(tags_detected) + " detected")
                continue


            image_pts = self.project_points(self.car_locations[image_index],
                                            self.target_locations[image_index])
            print(image_pts)


if __name__ == '__main__':
    projection = Projection('./target_cal/target1')
    projection.parse_location_data()



