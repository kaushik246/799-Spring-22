#! /usr/bin/python3

import vive_utils as vu
import numpy as np
import sys

filename = sys.argv[1]

f = open(filename,'r')

for line in f.readlines():
	values = line.split(',');
	values = values[12:]
	if len(values) > 12:
		values = values[0:12]
	mat = []
	for value in values:
		mat.append(float(value))
	mat = np.array(mat)
	mat = np.reshape(mat,(3,4))
	mat = np.append(mat,vu.mat_bottom,axis=0);
	point = np.matmul(mat,vu.origin_const);
	print("%0.8f,%0.8f,%0.8f" % (point[0][0],point[1][0],point[2][0]))
