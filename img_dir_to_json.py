import os
import cv2
import csv
import pickle
import numpy as np
from feature_extraction.haar_test import *

path = '/home/dungpb/test'

patient_path = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
patient_path.sort()

for pp in patient_path:
    csv_path = "testcsv/" + pp + ".csv"
    patient_image = os.path.join(path, pp, "images")
    print(patient_image)
    list_img = [f for f in os.listdir(patient_image) if os.path.isfile(os.path.join(patient_image, f))]
    list_img.sort()
    for i in range(len(list_img)):
        img_path = os.path.join(patient_image, list_img[i])
        img_data = cv2.imread(img_path, 0)
        with open(csv_path, 'a') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            wr.writerow(list(haar_extract(img_data, (256, 256))))