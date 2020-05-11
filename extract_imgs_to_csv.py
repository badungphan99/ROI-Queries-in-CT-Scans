import os
from feature_extraction.haar_test import *

import csv

label_dir = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn/data_label"
csv_path = "/home/dungpb/Work/HUS-AC/ROI-Queries-in-CT-Scans/dataset_features/train/11059.csv"


for i in [0,1,3,4,6,7,9]:
    print(i)
    label_path = os.path.join(label_dir, str(i))
    images_name = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
    count = 0
    for img in images_name:
        print(count)
        count+=1
        img_path = os.path.join(label_path, img)
        img_data = cv2.imread(img_path, 0)
        with open(csv_path, 'a') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            wr.writerow([str(i)] + list(haar_extract(img_data, (256, 256))))