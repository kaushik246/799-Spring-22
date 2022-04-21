#!/usr/bin/env python

from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
import numpy as np

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str,
                    default="DICT_6X6_250",
                    help="type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    ARUCO_DICT = {
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
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            args["type"]))
        sys.exit(0)

    print("[INFO] detecting '{}' tags...".format(args["type"]))
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
    arucoParams = cv2.aruco.DetectorParameters_create()

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    #cam_mat = np.array([[466.53298054, 0, 510.9976065, 0], [0, 475.71636522, 358.52360012, 0], [0, 0, 1, 0]])
    #inv_cam_mat = np.linalg.inv(cam_mat)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=1000)

        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
                                                           arucoDict, parameters=arucoParams)
        if len(corners) > 0:
            center_point_x = 0
            center_point_y = 0
            ids = ids.flatten()

            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                center_point_x += cX
                center_point_y += cY
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                cv2.putText(frame, "(" + str(cX) + " " + str(cY) + ")",
                    (topLeft[0], topLeft[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
            center_point_x = center_point_x/4.0
            center_point_y = center_point_y/4.0
            #three_d_points = np.dot(inv_cam_mat, [center_point_x, center_point_y, 1])
            #print(three_d_points)
            cv2.putText(frame, "(" + str(center_point_x) + " " + str(center_point_y) + ")",
                        (500, 500),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()