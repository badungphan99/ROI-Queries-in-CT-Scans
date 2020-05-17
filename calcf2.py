import json
import csv
import os
import cv2
import datetime
from feature_extraction.haar_test import *
import pickle

model_path = '/model/offcial/110518.sav'

json_path = "/home/dungpb/Work/HUS-AC/ROI-Queries-in-CT-Scans/annotations/datafullbody0206.json"
parent_dir = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/dataset"

regions = ["head", "chest", "abdominal"]

with open(json_path) as f:
    data = json.load(f)

sum = 0

for image in data["images"]:
    head_begin = -1
    head_end = -1
    chest_begin = -1
    chest_end = -1
    abdominal_begin = -1
    abdominal_end = -1
    num_img = -1
    patient_dir = ""
    if "patient_dir" in image:
        patient_dir = os.path.join(parent_dir, image["patient_dir"])
        print(patient_dir)
        num_img = len([f for f in os.listdir(patient_dir) if os.path.isfile(os.path.join(patient_dir, f))])
        print(num_img)
    else:
        print("ERROR: check path img in json file")

    for r in regions:
        if r in image:
            if "begin" in image[r]:
                if r == "head":
                    head_begin = image[r]["begin"]
                if r == "chest":
                    chest_begin = image[r]["begin"]
                if r == "abdominal":
                    abdominal_begin = image[r]["begin"]
            else:
                print("ERROR: no", r, "begin in", image["patient_dir"])
            if "end" in image["head"]:
                if r == "head":
                    head_end = image[r]["end"]
                if r == "chest":
                    chest_end = image[r]["end"]
                if r == "abdominal":
                    abdominal_end = image[r]["end"]
            else:
                print("ERROR: no", r, "end in", image["patient_dir"])

    img_path = [f for f in os.listdir(os.path.join(parent_dir, patient_dir)) if os.path.isfile(os.path.join(os.path.join(parent_dir, patient_dir), f))]
    img_path.sort()
    features = []

    for i in range(len(img_path)):
        img = cv2.imread(os.path.join(parent_dir, patient_dir, img_path[i]),0)
        img_size = 256
        start = datetime.datetime.now()
        features.append(list(tuyetvong(img, (256, 256))))
        print(datetime.datetime.now()-start)
        # exit(0)

    model = pickle.load(open(model_path, 'rb'))
    Y_hat = model.predict(features)
    counthead = 0
    countchest = 0
    countabdominal= 0
    for i in range(head_begin, head_end):
        if Y_hat[i] == 2:
            counthead += 1
    for i in range(chest_begin, chest_end):
        if Y_hat[i] == 5:
            countchest += 1
    for i in range(abdominal_begin, abdominal_end):
        if Y_hat[i] == 8:
            countabdominal += 1

    s = (counthead/(head_end-head_begin+1) + countchest/(chest_end-chest_begin+1) + countabdominal/(abdominal_end-abdominal_begin))/3
    print(s)
    sum = sum + s

print(sum/6)