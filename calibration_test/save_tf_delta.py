#! /usr/bin/python3
import sys
import numpy as np
import vive_utils as vu

name = sys.argv[1]
	
originTracker = "car1";
remoteTracker = "origin";

originLocation = vu.getTrackerLocationWeb(originTracker) 
remoteLocation = vu.getTrackerLocationWeb(remoteTracker) 

originMatrix = vu.lineTo4x3(originLocation);
remoteMatrix = vu.lineTo4x3(remoteLocation);

outputMatrix = vu.getDirectMatrix(originMatrix, np.identity(4), remoteMatrix , np.identity(4));
print(name + "," + vu.flatten4x4(outputMatrix));
