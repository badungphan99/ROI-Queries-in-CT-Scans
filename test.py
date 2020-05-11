import os
import cv2
import csv
import pickle
import numpy as np
from feature_extraction.haar_test import *

csv_path = "dataset_features/0511.csv"

path = '/home/dungpb/125Resize_offical'

patient_path = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
patient_path.sort()
count = 0
for pp in patient_path:
    count += 1
    patient_image = os.path.join(path, pp, "images")
    patient_bl = os.path.join(path, pp, "masks", "Bladder")
    patient_rt = os.path.join(path, pp, "masks", "Rectum")
    print(patient_image)
    list_img = [f for f in os.listdir(patient_image) if os.path.isfile(os.path.join(patient_image, f))]
    list_img.sort()
    img = [u[:len(u)-4] for u in list_img]

    list_bl = [f for f in os.listdir(patient_bl) if os.path.isfile(os.path.join(patient_bl, f))]
    list_bl.sort()
    bl = [u[:len(u)-4] for u in list_bl]

    list_rt = [f for f in os.listdir(patient_rt) if os.path.isfile(os.path.join(patient_rt, f))]
    list_rt.sort()
    rt = [u[:len(u)-4] for u in list_rt]
    # error /home/dungpb/125Resize_offical/Abdominal_2304_manual _Contours/images
    # get bl features
    if len(bl) >=4:
        for i in range(5):
            path_img = os.path.join(patient_image, bl[i]+".png")
            img_data = cv2.imread(path_img, 0)
            with open(csv_path, 'a') as csv_file:
                wr = csv.writer(csv_file, delimiter=',')
                wr.writerow([str(1)] + list(haar_extract(img_data, (256,256))))
    else:
        for i in bl:
            path_img = os.path.join(patient_image, str(i)+".png")
            img_data = cv2.imread(path_img, 0)
            with open(csv_path, 'a') as csv_file:
                wr = csv.writer(csv_file, delimiter=',')
                wr.writerow([str(1)] + list(haar_extract(img_data, (256, 256))))
    # get rt features
    if len(rt) >= 4:
        for i in range(5):
            path_img = os.path.join(patient_image, rt[i]+".png")
            img_data = cv2.imread(path_img, 0)
            with open(csv_path, 'a') as csv_file:
                wr = csv.writer(csv_file, delimiter=',')
                wr.writerow([str(2)] + list(haar_extract(img_data, (256, 256))))
    else:
        for i in rt:
            path_img = os.path.join(patient_image, str(i)+".png")
            img_data = cv2.imread(path_img, 0)
            with open(csv_path, 'a') as csv_file:
                wr = csv.writer(csv_file, delimiter=',')
                wr.writerow([str(2)] + list(haar_extract(img_data, (256, 256))))

    #get non bl and rt features
    non_bl_rt = []
    start = int(bl[0]) if int(bl[0]) < int(rt[0]) else int(rt[0])
    end = int(bl[len(bl)-1]) if int(bl[len(bl)-1]) > int(rt[len(rt)-1]) else int(rt[len(rt)-1])

    start_img = int(img[0])
    end_img = int(img[len(img)-1])

    if start <= end-3:
        non_bl_rt.extend(np.random.randint(start, end, 3))
        if start_img <= start - 1:
            non_bl_rt.extend(np.random.randint(start_img, start, 1))
            if end <= end_img - 1:
                non_bl_rt.extend(np.random.randint(end, end_img, 1))
        else:
            if end <= end_img - 2:
                non_bl_rt.extend(np.random.randint(end, end_img, 2))
    else:
        if start_img <= start - 3:
            non_bl_rt.extend(np.random.randint(start_img, start, 3))
            if end <= end_img - 2:
                non_bl_rt.extend(np.random.randint(end, end_img, 2))
        else:
            if end <= end_img - 5:
                non_bl_rt.extend(np.random.randint(end, end_img, 5))

    for i in non_bl_rt:
        path_img = os.path.join(patient_image, str(i)+".png")
        img_data = cv2.imread(path_img, 0)
        with open(csv_path, 'a') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            wr.writerow([str(3)] + list(haar_extract(img_data, (256, 256))))

    print(count)