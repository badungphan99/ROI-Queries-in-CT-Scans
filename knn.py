import pandas
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
import pickle
from sklearn.svm import SVC



filename = '11058.csv'

datatrain = pandas.read_csv('dataset_features/train/'+filename, header=None)
X = datatrain[datatrain.columns[1:datatrain.shape[1]]].to_numpy()
Y = datatrain[0].to_numpy()

# datatest = pandas.read_csv('dataset_features/test/'+filename, header=None)
# X_test = datatest[datatest.columns[1:datatest.shape[1]]].to_numpy()
# Y_test = datatest[0].to_numpy()

size_test = 0.3 # test / total

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=round(len(X)*size_test))

# clf = SVC(kernel='precomputed', gamma='auto')
# clf.fit(X_train, Y_train)
#
#
model = neighbors.KNeighborsClassifier(n_neighbors= 7, p = 1)
model.fit(X_train, Y_train)
#
print("Accuracy : %.2f %%" %(100*model.score(X_test, Y_test)))
print("F1 : %.2f %%" %(100*metrics.f1_score(y_true=Y_test, y_pred=model.predict(X_test), average='weighted')))
# filename = 'model/05115.sav'
# with open(filename, 'wb') as file:
#     pickle.dump(model, file)

# load = pickle.load(open(filename,'rb'))
# y_p = model.pr

# data1 = pandas.read_csv('xxx.csv', header=None)
# x_val = data1[data1.columns[0:data1.shape[1]]].to_numpy()
# y_val = model.predict(x_val)
# print(y_val)

# print(len(x_val[0]))