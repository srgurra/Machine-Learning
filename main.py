import pandas as pd
import numpy as np

por = pd.read_csv("student-por.csv")
mat = pd.read_csv("student-mat.csv")
mat['Class'] = "math"
por['Class'] = "port"


data = pd.concat([mat,por],axis = 0)

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

target = data['G3']
data.drop('G3',axis=1,inplace=True)

for col in ['school','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery',
           'higher','internet','romantic','Class','sex','address','famsize','Medu','Fedu']:
    le = LabelEncoder()
    le.fit(data[col])
    data[col] = le.transform(data[col])
    

#linear regression
trainX,testX,trainY,testY = train_test_split(data,target,test_size = 0.2,random_state=12)
lm = LinearRegression()
lm.fit(trainX,trainY)
pred = lm.predict(testX)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(testY,pred))


#k nearest neighbor
from sklearn.neighbors import KNeighborsRegressor
trainX,testX,trainY,testY = train_test_split(data,target,test_size = 0.2,random_state=12)
neigh = KNeighborsRegressor(p=2)
neigh.fit(trainX,trainY)

pred = neigh.predict(testX)
print(mean_squared_error(testY,pred))

#SVM regressor
from sklearn.svm import SVR

svr = SVR(C=0.1,epsilon=1)
svr.fit(trainX,trainY)
pred = svr.predict(testX)
#print(mean_squared_error(trainY,svr.predict(trainX)))
print(mean_squared_error(testY,pred))

#prints the prediction score for different model parameters
scores = []
for c in range(1,20,2):
    for eps in range(1,100,10):
        svr = SVR(C=float(c/10),epsilon=float(eps/100))
        svr.fit(trainX,trainY)
        pred = svr.predict(testX)
        
        scores.append(mean_squared_error(testY,pred))


#neural network
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(100,50,), activation = "relu", solver = "adam", alpha = 0.001)
mlp.fit(trainX,trainY)
pred = mlp.predict(testX)

print(mean_squared_error(testY,pred))
