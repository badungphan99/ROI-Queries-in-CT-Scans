import pandas
from sklearn import neighbors
import pickle
from feature_extraction.haar_test import *
import os

path = '/mnt/32D84D55D84D188D/ubuntu-data-hdd/dataset/FULL_001/crop'
model_path = '/home/dungpb/Work/HUS-AC/ROI-Queries-in-CT-Scans/model/05115.sav'

img = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
img.sort()
X_test = []

for i in range(len(img)):
    print(i)
    img_data = cv2.imread(os.path.join(path, img[i]),0)
    X_test.append(haar_extract(img_data,(256,256)))

model = pickle.load(open(model_path,'rb'))
print(model.predict(X_test))