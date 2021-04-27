# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor

data = pd.read_csv('jsonalldata.csv')
data.describe()

finaldata = data.drop(labels = ["road"],axis = 1) 
finaldata.head(5)

Y = finaldata["AADT"]
X = finaldata.drop(labels = ["AADT"],axis = 1)

print(Y.head(5))

    
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1, random_state=2)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train = quantile_transformer.fit_transform(X_train)
X_val = quantile_transformer.transform(X_val)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)

model=Sequential()
model.add(Dense(16,input_dim=2,activation='relu'))
#model.add(Dense(32,activation='relu'))
#model.add(Dense(64,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
#Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=100,batch_size=100)

#evaluate the model
training=model.evaluate(X_train,Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1],training[1]*100))

pred = pd.read_csv('jsonclean.csv')
pred = preprocessing.StandardScaler().fit(pred)
X_pred = pred.drop(labels = ["road"],axis = 1) 
print(X_pred.head(5))

Y_pred = model.predict(X_pred, verbose = 0)

pd.DataFrame(Y_pred).to_csv("Y_pred.csv")

print(Y_pred)

df = pd.read_csv('./Comparison.csv').values
o = df[:,0]
m = df[:,1]
n = df[:,2]
plt.plot(o,m,"o",color='orange')
plt.plot(o,n,"o",color='blue')
plt.xlabel("Road points")
plt.ylabel("AADT values")
plt.legend(['Base', 'Predicted']);
plt.show()
