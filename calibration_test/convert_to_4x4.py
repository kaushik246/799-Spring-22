#! /usr/bin/python3

import sys
import vive_utils as vu


filename = sys.argv[1];
f=open(filename,'r');

for line in f.readlines():
	firstComma = line.find(',');
	lineTitle = line[0:firstComma]
	restOfLine = line[firstComma+1:]
	relativeMatrix = vu.lineTo4x4(restOfLine)
	coords = vu.getTranslationFrom4x4(relativeMatrix)
	print(lineTitle)
	print(coords)
