import pandas
from sklearn import neighbors
import pickle
from feature_extraction.haar_test import *
import os

path = '/mnt/32D84D55D84D188D/ubuntu-data-hdd/dataset/FULL_002/crop'
model_path = '/model/offcial/110518.sav'

img = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
img.sort()
X_test = []

for i in range(len(img)):
    img_data = cv2.imread(os.path.join(path, img[i]),0)
    X_test.append(tuyetvong(img_data,(256,256)))

model = pickle.load(open(model_path,'rb'))
res = model.predict(X_test)
for i in range(270,296):
    print(res[i])