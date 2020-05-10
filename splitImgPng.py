import csv
import os
from feature_extraction.haar_test import *
import numpy as np
import datetime

label_0 = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn/data_label/0"

label_1 = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn/data_label/1"

label_3 = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn/data_label/3"

label_4 = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn/data_label/4"

label_6 = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn/data_label/6"

label_7 = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn/data_label/7"

label_9 = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn/data_label/9"
# csv_path = "dataset_features/train/haar-knn-test.csv"

root_path = "/mnt/32D84D55D84D188D/ubuntu-data-hdd/data-for-haar-knn"

head_dataset = os.path.join(root_path, "head")
chest_dataset = os.path.join(root_path, "chest")
abdominal_dataset = os.path.join(root_path, "abdominal")

count = 0

# Head processing
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
        imgwrite = os.path.join(label_1, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
        cv2.imwrite(imgwrite, img_data)
        # with open(csv_path, 'a') as csv_file:
        #     wr = csv.writer(csv_file, delimiter=',')
        #     wr.writerow([str(1)] + list(haar_extract(img_data, (256, 256))))
    # lay 5 anh cuoi cung co brain
    for i in range(len(list_br)-5,len(list_br)):
        path_img = os.path.join(head_images, br[i] + ".png")
        img_data = cv2.imread(path_img, 0)
        imgwrite = os.path.join(label_3, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
        cv2.imwrite(imgwrite, img_data)
        # with open(csv_path, 'a') as csv_file:
        #     wr = csv.writer(csv_file, delimiter=',')
        #     wr.writerow([str(3)] + list(haar_extract(img_data, (256, 256))))

    # lay 5 anh bat kia giua khoang bat dau va ket thuc cua brain
    ran_img = np.random.randint(low=int(br[5]),high=int(br[len(br)-5]),size=5)
    for r in ran_img:
        path_img = os.path.join(head_images, str(r).zfill(3) + ".png")
        img_data = cv2.imread(path_img, 0)
        imgwrite = os.path.join(label_0, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
        cv2.imwrite(imgwrite, img_data)
        # with open(csv_path, 'a') as csv_file:
        #     wr = csv.writer(csv_file, delimiter=',')
        #     wr.writerow([str(0)] + list(haar_extract(img_data, (256, 256))))

    # neu tong so luong anh cua tap data > so luong mask brain + 30 lay ngau nhien 5 anh trong khoang nay
    if int(list_img[len(list_img)-1][:len(list_img[len(list_img)-1])-4]) > int(br[len(br)-1]) + 30:
        ran_img = np.random.randint(low=int(br[len(br)-1]) + 30, high=int(list_img[len(list_img)-1][:len(list_img[len(list_img)-1])-4]),size=5)
        for r in ran_img:
            path_img = os.path.join(head_images, str(r).zfill(3) + ".png")
            img_data = cv2.imread(path_img, 0)
            imgwrite = os.path.join(label_0, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
            cv2.imwrite(imgwrite, img_data)
            # with open(csv_path, 'a') as csv_file:
            #     wr = csv.writer(csv_file, delimiter=',')
            #     wr.writerow([str(0)] + list(haar_extract(img_data, (256, 256))))

# Chest processing
chest_path = [f for f in os.listdir(chest_dataset) if os.path.isdir(os.path.join(chest_dataset, f))]
for cpp in chest_path:
    print(cpp)
    chest_images = os.path.join(chest_dataset, cpp, "crop")
    chest_lung_l = os.path.join(chest_dataset, cpp, "masksOrg", "Left Lung")
    chest_lung_r = os.path.join(chest_dataset, cpp, "masksOrg", "Right Lung")
    chest_sc = os.path.join(chest_dataset, cpp, "masksOrg", "Spinal Canal")

    list_ll = [f for f in os.listdir(chest_lung_l) if os.path.isfile(os.path.join(chest_lung_l, f))]
    list_lr = [f for f in os.listdir(chest_lung_r) if os.path.isfile(os.path.join(chest_lung_r, f))]
    list_sc = [f for f in os.listdir(chest_sc) if os.path.isfile(os.path.join(chest_sc, f))]

    list_ll.sort()
    list_lr.sort()
    list_sc.sort()

    ll=[u[:len(u)-4] for u in list_ll]
    lr = [u[:len(u) - 4] for u in list_lr]
    sc = [u[:len(u) - 4] for u in list_sc]

    start = min(int(ll[0]),int(lr[0]))
    end = max(int(ll[len(ll) - 1]), int(lr[len(lr) - 1]))

    for i in range(start, start+5):
        path_img = os.path.join(chest_images, str(i).zfill(3) + ".png")
        img_data = cv2.imread(path_img, 0)
        imgwrite = os.path.join(label_4, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
        cv2.imwrite(imgwrite, img_data)

    for i in range(end-5, end):
        path_img = os.path.join(chest_images, str(i).zfill(3) + ".png")
        img_data = cv2.imread(path_img, 0)
        imgwrite = os.path.join(label_6, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
        cv2.imwrite(imgwrite, img_data)

    ran_img = np.random.randint(low=start+5, high=end-6, size=5)
    for r in ran_img:
        path_img = os.path.join(chest_images, str(r).zfill(3) + ".png")
        img_data = cv2.imread(path_img, 0)
        imgwrite = os.path.join(label_0, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
        cv2.imwrite(imgwrite, img_data)

    if int(sc[len(sc)-1]) > (end + 5):
        ran_img = np.random.randint(low=end, high=int(sc[len(sc)-1]), size=5)
        for r in ran_img:
            path_img = os.path.join(chest_images, str(r).zfill(3) + ".png")
            img_data = cv2.imread(path_img, 0)
            imgwrite = os.path.join(label_0, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
            cv2.imwrite(imgwrite, img_data)


# Abdomianl processing
abdominal_path = [f for f in os.listdir(abdominal_dataset) if os.path.isdir(os.path.join(abdominal_dataset, f))]
for app in abdominal_path:
    print(app)
    abdominal_images = os.path.join(abdominal_dataset, app, "images")
    abdominal_bladder = os.path.join(abdominal_dataset, app, "masks", "Bladder")
    abdominal_lf = os.path.join(abdominal_dataset, app, "masks", "Left Femur")
    abdominal_rf = os.path.join(abdominal_dataset, app, "masks", "Right Femur")
    abdominal_rt = os.path.join(abdominal_dataset, app, "masks", "Rectum")

    list_bl = [f for f in os.listdir(abdominal_bladder) if os.path.isfile(os.path.join(abdominal_bladder, f))]
    list_lf = [f for f in os.listdir(abdominal_lf) if os.path.isfile(os.path.join(abdominal_lf, f))]
    list_rf = [f for f in os.listdir(abdominal_rf) if os.path.isfile(os.path.join(abdominal_rf, f))]
    list_rt = [f for f in os.listdir(abdominal_rt) if os.path.isfile(os.path.join(abdominal_rt, f))]

    list_bl.sort()
    list_lf.sort()
    list_rf.sort()
    list_rt.sort()

    bl = [u[:len(u) - 4] for u in list_bl]
    lf = [u[:len(u) - 4] for u in list_lf]
    rf = [u[:len(u) - 4] for u in list_rf]
    rt = [u[:len(u) - 4] for u in list_rt]

    start = min(int(bl[0]), int(lf[0]), int(rf[0]), int(rt[0]))
    end = max(int(bl[len(bl)-1]), int(lf[len(lf)-1]), int(rf[len(rf)-1]), int(rt[len(rt)-1]))

    for i in range(start, start+5):
        path_img = os.path.join(abdominal_images, str(i) + ".png")
        img_data = cv2.imread(path_img, 0)
        imgwrite = os.path.join(label_7, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
        cv2.imwrite(imgwrite, img_data)

    for i in range(end - 5, end):
        path_img = os.path.join(abdominal_images, str(i) + ".png")
        img_data = cv2.imread(path_img, 0)
        imgwrite = os.path.join(label_9, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
        cv2.imwrite(imgwrite, img_data)
    if start + 5 < end - 6:
        ran_img = np.random.randint(low=start + 5, high=end - 6, size=5)
        for r in ran_img:
            path_img = os.path.join(abdominal_images, str(r) + ".png")
            img_data = cv2.imread(path_img, 0)
            imgwrite = os.path.join(label_0, datetime.datetime.now().strftime("%d-%m-%Y:%H:%M:%S:%f") + ".png")
            cv2.imwrite(imgwrite, img_data)