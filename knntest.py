import pandas
from sklearn import neighbors
import pickle
import os

path = 'testcsv'
result = 'result'

patient_path = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for pp in patient_path:
    ppcsv = os.path.join(path,pp)
    data = pandas.read_csv(ppcsv, header=None)
    x_test = data.to_numpy()
    filename = 'model/bl_rt.sav'
    model = pickle.load(open(filename, 'rb'))
    ppresult = os.path.join(result, pp+".txt")
    f = open(ppresult, "w")
    f.write(str(model.predict(x_test)))
    f.close()