import json
import csv
import cv2
import os
from haarlikefeature import *

def get_features(patient_dir_path: str, patient, size: int, features_descriptions):
    features = []
    path = os.path.join(patient_dir_path, patient.get_parent_dir()) + "/crop"
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()
    print(path)
    for i in range(0, len(files)):
        if(patient.get_index_begin() <= i and i <= patient.get_index_end()):
            print(os.path.join(path, files[i]))
            image = cv2.imread(os.path.join(path, files[i]), 0)
            image = cv2.resize(image, (size, size))
            haarlike = HaarlikeFeature()
            features.append(haarlike.extractFeatures(image, features_descriptions))

    return features


def data_loader(patient_dir_path: str, json_path: str, size):
    info = []

    with open(json_path) as f:
        data = json.load(f)

    for region in data.keys():
        for e in data[region]['image']:
            r = Region(e['parent_dir'], e['begin'], e['end'], region)
            info.append(r)

    haarlike = HaarlikeFeature()
    features_cnt, descriptions = haarlike.determineFeatures(size,size)
    features_descriptions = descriptions[::-1]

    for i in info:
        features = get_features(patient_dir_path, i, size, features_descriptions)
        i.set_features(features)

    return info

class Region:
    def __init__(self, parent_dir: str, begin: int, end: int, region: str):
        self.__parent_dir = parent_dir
        self.__begin = begin
        self.__end = end
        self.__region = region
        self.__features = []
        self.__features_begin = []
        self.__features_end = []

    def __str__(self):
        return self.__parent_dir + " " + str(self.__begin) + " " + str(self.__end) + " " + self.__region

    def set_features(self, features: []):
        self.__features = features
        self.__features_begin = features[0:5]
        self.__features_end = features[len(features)-5:len(features)]

    def get_parent_dir(self):
        return self.__parent_dir

    def get_index_begin(self):
        return self.__begin

    def get_index_end(self):
        return self.__end

    def save_csv(self, path: str):
        region = 0
        region_begin = 0.1
        region_end = 0.2
        if(self.__region == 'head'):
            region = 1
            region_begin = 1.1
            region_end = 1.2
        if(self.__region == "chest"):
            region = 2
            region_begin = 2.1
            region_end = 2.2
        if(self.__region == "abdominal"):
            region = 3
            region_begin = 3.1
            region_end = 3.2
        with open(path, 'a') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            for feature in self.__features:
                wr.writerow([str(region)] + list(feature))
            for feature_begin in self.__features_begin:
                wr.writerow([str(region_begin)] + list(feature_begin))
            for feature_end in self.__features_end:
                wr.writerow([str(region_end)] + list(feature_end))

if __name__ == "__main__" :
    parent = '/home/dungpb/dataset'

    path = '/home/dungpb/Work/HUS-AC/haar-knn/datatestfullbody0206.json'

    csvpath = '/home/dungpb/dataset/features0206-64.csv'

    info = data_loader(parent, path, 64)

    for i in info:
        i.save_csv(csvpath)

