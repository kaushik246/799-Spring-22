import csv 
import sys

def bb_intersection_over_union(boxA, boxB):
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

def generate_dict(dictionary, file1):

    with open(file1, newline='') as text1:
        file1reader = csv.reader(text1, delimiter=',')

        for row in file1reader:
            img_name = row[0]
            x_min = float(row[1])
            y_min = float(row[2])
            x_max = float(row[3])
            y_max = float(row[4])

            box1 = [x_min, y_min, x_max, y_max]

            if img_name not in dictionary:
                dictionary[img_name] = []

            dictionary[img_name].append(box1)


if __name__=='__main__':
 
   dictionary = {}
   file1 = sys.argv[1]
   file2 = sys.argv[2]
   
   generate_dict(dictionary, file1)
   generate_dict(dictionary, file2)

   log_file = open('log.txt', mode='w')
   output_file = open('output_metrics.txt', mode='w')

   for key in dictionary:

       if(len(dictionary[key]) != 2):
           log_file.write(key+ "\n")
       else:

           bbox1 = dictionary[key][0]
           bbox2 = dictionary[key][1]
           iou = bb_intersection_over_union(bbox1, bbox2)
           output_file.write(key + ", " + str(iou) + "\n")
    

         

