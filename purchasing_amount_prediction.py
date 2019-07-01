#Predicting customer purchase amount

import numpy as np
import pandas as pd

dataset=pd.read_csv('train.csv')
X=dataset.iloc[10000,[0,1,2,3,4,5,6,7,8,9,10]].values
Y=dataset.iloc[10000,-1].values

from sklearn.preprocessing import LabelEncoder
lab_enc=LabelEncoder()
X[10000,1]=lab_enc.fit_transform(X[10000,1])
X[10000,3]=lab_enc.fit_transform(X[10000,3])
X[10000,6]=lab_enc.fit_transform(X[10000,6])
X[10000,2]=lab_enc.fit_transform(X[10000,2])
X[10000,5]=lab_enc.fit_transform(X[10000,5])
#Z=pd.DataFrame(X)

from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp=imp.fit(X[10000,9:11])
X[10000,9:11]=imp.fit_transform(X[10000,9:11])


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
#Z=sc.fit_transform(Z)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100)
rf.fit(X,Y)

#test set
dataset1=pd.read_csv('test.csv')
X_test=dataset1.iloc[10000,[0,1,2,3,4,5,6,7,8,9,10]].values

Y_test=rf.predict(X_test)

