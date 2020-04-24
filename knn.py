import pandas
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import pickle

data = pandas.read_csv('dataset_features/test.csv', header=None)
X = data[data.columns[1:data.shape[1]]].to_numpy()
Y = data[0].to_numpy()

size_test = 0.3 # test / total

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=round(len(X)*size_test))

model = neighbors.KNeighborsClassifier(n_neighbors= 4)
model.fit(X_train, Y_train)

print("Accuracy : %.2f %%" %(100*model.score(X_test, Y_test)))

filename = 'model/test01.sav'
with open(filename, 'wb') as file:
    pickle.dump(model, file)