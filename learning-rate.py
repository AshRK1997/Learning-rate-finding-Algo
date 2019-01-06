import pandas as pd
import numpy as np


a = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
                names=["a", "b", "c", "d", "e", "f", "g", "h", "i"])

b = a

c = b.filter(a.columns[[8]], axis=1)
a.drop(a.columns[[8]], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le1.fit(a.a)
a.a = le1.transform(a.a)

from sklearn.preprocessing import OneHotEncoder

x = np.array(a)
y = np.array(c)

ohe = OneHotEncoder(categorical_features=[0])

ohe.fit(x)

x = ohe.transform(x).toarray()

from sklearn.model_selection import train_test_split

xtr, xts, ytr, yts = train_test_split(x, y, test_size=0.2)

from xgboost import XGBRegressor

e = 0.1
sr = [0]
f=e
while(len(str(e)) < 20):


    model4 = XGBRegressor(learning_rate=e)
    model4.fit(xtr, ytr)
    g = float("{0:.25f}".format(model4.score(xts, yts)))
    sr.append(g)
    print("score,e=,f=",g,e,f)
    length = len(sr)
    print("sr len = ",length)
    print("sr = ",sr)
    if sr[length-1]<sr[length-2]:
        e = e-f
        f = f*0.1
        e = e+f
    else:
        e = e+f
    print("e=", e)







