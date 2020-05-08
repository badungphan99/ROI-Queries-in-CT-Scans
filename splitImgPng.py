import csv
import os
from feature_extraction.haar_test import *
import numpy as np

csv_path = "dataset_features/train/haar-knn-test.csv"

root_path = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn"

head_dataset = os.path.join(root_path, "head")
chest_dataset = os.path.join(root_path, "chest")
abdominal_dataset = os.path.join(root_path, "abdominal")

#Head processing
head_path = [f for f in os.listdir(head_dataset) if os.path.isdir(os.path.join(head_dataset, f))]
for hpp in head_path:
    print(hpp)
    head_images = os.path.join(head_dataset, hpp, "crop")
    head_brain = os.path.join(head_dataset, hpp, "masksOrg", "Brain")

    list_img = [f for f in os.listdir(head_images) if os.path.isfile(os.path.join(head_images, f))]

    list_br = [f for f in os.listdir(head_brain) if os.path.isfile(os.path.join(head_brain, f))]
    list_br.sort()
    br = [u[:len(u)-4] for u in list_br]
    # lay 5 anh dau tien co brain
    for i in range(5):
        path_img = os.path.join(head_images,br[i]+".png")
        img_data = cv2.imread(path_img, 0)
        with open(csv_path, 'a') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            wr.writerow([str(1)] + list(haar_extract(img_data, (256, 256))))
    # lay 5 anh cuoi cung co brain
    for i in range(len(list_br)-5,len(list_br)):
        path_img = os.path.join(head_images, br[i] + ".png")
        img_data = cv2.imread(path_img, 0)
        with open(csv_path, 'a') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            wr.writerow([str(3)] + list(haar_extract(img_data, (256, 256))))

    # lay 5 anh bat kia giua khoang bat dau va ket thuc cua brain
    ran_img = np.random.randint(low=int(br[5]),high=int(br[len(br)-5]),size=5)
    for r in ran_img:
        path_img = os.path.join(head_images, str(r).zfill(3) + ".png")
        img_data = cv2.imread(path_img, 0)
        with open(csv_path, 'a') as csv_file:
            wr = csv.writer(csv_file, delimiter=',')
            wr.writerow([str(0)] + list(haar_extract(img_data, (256, 256))))

    # neu tong so luong anh cua tap data > so luong mask brain + 30 lay ngau nhien 5 anh trong khoang nay
    if int(list_img[len(list_img)-1][:len(list_img[len(list_img)-1])-4]) > int(br[len(br)-1]) + 30:
        ran_img = np.random.randint(low=int(br[len(br)-1]) + 30, high=int(list_img[len(list_img)-1][:len(list_img[len(list_img)-1])-4]),size=5)
        for r in ran_img:
            path_img = os.path.join(head_images, str(r).zfill(3) + ".png")
            img_data = cv2.imread(path_img, 0)
            with open(csv_path, 'a') as csv_file:
                wr = csv.writer(csv_file, delimiter=',')
                wr.writerow([str(0)] + list(haar_extract(img_data, (256, 256))))

#Chest processing
chest_path = [f for f in os.listdir(chest_dataset) if os.path.isdir(os.path.join(chest_dataset, f))]
for cpp in chest_path:
    pass