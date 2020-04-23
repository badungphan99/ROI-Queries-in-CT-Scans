from haarlikefeature import *
import cv2
import timeit

image = cv2.imread('/home/dungpb/Work/HUS-AC/haar-knn/CT_Image_Storage-ABD_003__-Patient_Model--.png',0)

image = cv2.resize(image, (32,32))

height, width = image.shape

start = timeit.default_timer()

haarlike = HaarlikeFeature()

print(width, " ", height)

features_cnt, descriptions = haarlike.determineFeatures(32, 32)

stop1 = timeit.default_timer()

print('Time 1 : ', stop1 - start)

features_descriptions = descriptions[::-1]

features = haarlike.extractFeatures(image, features_descriptions)

stop = timeit.default_timer()

print('Time 2 : ', stop - stop1)

print(len(features))