import pandas
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import pickle

filename = '02063.csv'

datatrain = pandas.read_csv('dataset_features/train/'+filename, header=None)
X_train = datatrain[datatrain.columns[1:datatrain.shape[1]]].to_numpy()
Y_train = datatrain[0].to_numpy()

datatest = pandas.read_csv('dataset_features/test/'+filename, header=None)
X_test = datatest[datatest.columns[1:datatest.shape[1]]].to_numpy()
Y_test = datatest[0].to_numpy()

model = neighbors.KNeighborsClassifier(n_neighbors= 7, p = 2)
model.fit(X_train, Y_train)

print("Accuracy : %.2f %%" %(100*model.score(X_test, Y_test)))

# filename = 'model/bl_rt.sav'
# # with open(filename, 'wb') as file:
# #     pickle.dump(model, file)
#
# load = pickle.load(open(filename,'rb'))
# y_p = model.pr

# data1 = pandas.read_csv('xxx.csv', header=None)
# x_val = data1[data1.columns[0:data1.shape[1]]].to_numpy()
# y_val = model.predict(x_val)
# print(y_val)

# print(len(x_val[0]))