import json
import csv
import os
import cv2
from feature_extraction.haar_test import *
import pickle

# mode : 0 - lay cac gia tri theo vung dau, bung, nguc
# mode : 1 - lay cac gia tri theo bat dau va ket thuc cua moi vung
def extract_csv(parent_dir_path: str, json_path: str, features_out_path: str, mode: int = 1):
    regions = ["head", "chest", "abdominal"]

    with open(json_path) as f:
        data = json.load(f)

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
            patient_dir = os.path.join(parent_dir_path, image["patient_dir"])
            print(patient_dir)
            num_img = len([f for f in os.listdir(patient_dir) if os.path.isfile(os.path.join(patient_dir, f))])
            print(num_img)
        else:
            print("ERROR: check path img in json file")
            return False

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
                    return False
                if "end" in image["head"]:
                    if r == "head":
                        head_end = image[r]["end"]
                    if r == "chest":
                        chest_end = image[r]["end"]
                    if r == "abdominal":
                        abdominal_end = image[r]["end"]
                else:
                    print("ERROR: no", r, "end in", image["patient_dir"])
                    return False

        patient = Patient(patient_dir, num_img, head_begin, head_end, chest_begin, chest_end, abdominal_begin, abdominal_end)
        patient.init_features()
        patient.write_csv(features_out_path, mode)


class Patient:
    def __init__(self, patient_dir: str, num_img: int, head_begin: int, head_end: int, chest_begin: int, chest_end: int,
                 abdominal_begin: int, abdominal_end: int):
        self.__patient_dir = patient_dir
        self.__num_img = num_img
        self.__head_begin = head_begin
        self.__head_end = head_end
        self.__chest_begin = chest_begin
        self.__chets_end = chest_end
        self.__abdominal_begin = abdominal_begin
        self.__abdominal_end = abdominal_end
        self.__features = []
        self.__region = []

    def init_features(self):
        img_path = [f for f in os.listdir(self.__patient_dir) if os.path.isfile(os.path.join(self.__patient_dir, f))]
        img_path.sort()
        for i in range(0,self.__num_img - 1):
            img = cv2.imread(os.path.join(self.__patient_dir, img_path[i]), 0)
            img_size = 256
            self.__features.append(list(tuyetvong(img, (img_size, img_size))))
            # print(list(tuyetvong(img, (img_size, img_size))))

    def write_csv(self, csv_path: str, mode: int):
        for i in range(0, self.__num_img - 1):
            region = 0

            if self.__head_begin <= i <= self.__head_end:
                region = 2
            if self.__chest_begin <= i <= self.__chets_end:
                region = 5
            if self.__abdominal_begin <= i <= self.__abdominal_end:
                region = 8
            self.__region.append(region)
        model_path = '/model/offcial/110518.sav'
        model = pickle.load(open(model_path, 'rb'))
        count = 0
        Y_pre = model.predict(self.__features)
        for i in range(0, self.__num_img - 1):
            if Y_pre[i] == self.__region[i]:
                count += 1

        print(count/self.__num_img)

        # with open(csv_path, 'a') as csv_file:
        #     wr = csv.writer(csv_file, delimiter=',')
        #     countf2 = 0
        #     count = 0


            #     if mode == 0:
            #         # wr.writerow([str(region)] + list(self.__features[i]))
            #         # print(list(self.__features[i]))
            #         X_test = []
            #         X_test.append(self.__features[i])
            #         if region == model.predict(X_test)[0]:
            #             countf2 += 1
            #
            #     if mode == 1 :
            #         if region == 0 and count < 6:
            #             wr.writerow([str(region)] + list(self.__features[i]))
            #             count += 1
            #         if self.__head_begin <= i <= self.__head_begin + 5:
            #             wr.writerow([str(1)] + list(self.__features[i]))
            #         if self.__head_end - 5 <= i <= self.__head_end:
            #             wr.writerow([str(3)] + list(self.__features[i]))
            #         if self.__chest_begin <= i <= self.__chest_begin + 5:
            #             wr.writerow([str(4)] + list(self.__features[i]))
            #         if self.__chets_end - 5 <= i <= self.__chets_end:
            #             wr.writerow([str(6)] + list(self.__features[i]))
            #         if self.__abdominal_begin <= i <= self.__abdominal_begin + 5:
            #             wr.writerow([str(7)] + list(self.__features[i]))
            #         if self.__abdominal_end - 5 <= i <= self.__abdominal_end:
            #             wr.writerow([str(9)] + list(self.__features[i]))





if __name__ == "__main__":
    parent_dir = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/dataset"
    json_path = "/home/dungpb/Work/HUS-AC/ROI-Queries-in-CT-Scans/annotations/datafullbody0206.json"
    csv_path = "dataset_features/test/02062342343.csv"

    extract_csv(parent_dir, json_path, csv_path, 0)
