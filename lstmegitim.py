# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:50:25 2021

@author: Mahmut Arslan
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.recurrent import LSTM
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import math


data1 = pd.read_csv("B05_birlestirilmis.csv")
data2 = pd.read_csv("B07_birlestirilmis.csv")
data3 = pd.read_csv("B18_birlestirilmis.csv")
data4 = pd.read_csv("B33_birlestirilmis.csv")
data5 = pd.read_csv("B34_birlestirilmis.csv")
data6 = pd.read_csv("B46_birlestirilmis.csv")
data7 = pd.read_csv("B47_birlestirilmis.csv")
data8 = pd.read_csv("B48_birlestirilmis.csv")

X1=data1.iloc[:,0:31]
Y1=data1.iloc[:,30:31]
X2=data2.iloc[:,0:31]
Y2=data2.iloc[:,30:31]
X3=data3.iloc[:,0:31]
Y3=data3.iloc[:,30:31]
X4=data4.iloc[:,0:31]
Y4=data4.iloc[:,30:31]
X5=data5.iloc[:,0:31]
Y5=data5.iloc[:,30:31]
X6=data6.iloc[:,0:31]
Y6=data6.iloc[:,30:31]
X7=data7.iloc[:,0:31]
Y7=data7.iloc[:,30:31]
X8=data8.iloc[:,0:31]
Y8=data8.iloc[:,30:31]

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
trX1, teX1,trY1,teY1 = train_test_split(X1,Y1,test_size=0.20, random_state=0)
trX2, teX2,trY2,teY2 = train_test_split(X2,Y2,test_size=0.20, random_state=0)
trX3, teX3,trY3,teY3 = train_test_split(X3,Y3,test_size=0.20, random_state=0)
trX4, teX4,trY4,teY4 = train_test_split(X4,Y4,test_size=0.20, random_state=0)
trX5, teX5,trY5,teY5 = train_test_split(X5,Y5,test_size=0.20, random_state=0)
trX6, teX6,trY6,teY6 = train_test_split(X6,Y6,test_size=0.20, random_state=0)
trX7, teX7,trY7,teY7 = train_test_split(X7,Y7,test_size=0.20, random_state=0)
trX8, teX8,trY8,teY8 = train_test_split(X8,Y8,test_size=0.20, random_state=0)

tesX1=pd.DataFrame(teX1).sort_index()
tesY1=pd.DataFrame(teY1).sort_index()
tesX2=pd.DataFrame(teX2).sort_index()
tesY2=pd.DataFrame(teY2).sort_index()
tesX3=pd.DataFrame(teX3).sort_index()
tesY3=pd.DataFrame(teY3).sort_index()
tesX4=pd.DataFrame(teX4).sort_index()
tesY4=pd.DataFrame(teY4).sort_index()
tesX5=pd.DataFrame(teX5).sort_index()
tesY5=pd.DataFrame(teY5).sort_index()
tesX6=pd.DataFrame(teX6).sort_index()
tesY6=pd.DataFrame(teY6).sort_index()
tesX7=pd.DataFrame(teX7).sort_index()
tesY7=pd.DataFrame(teY7).sort_index()
tesX8=pd.DataFrame(teX8).sort_index()
tesY8=pd.DataFrame(teY8).sort_index()


trainX1=pd.DataFrame(trX1).sort_index()
trainY1=pd.DataFrame(trY1).sort_index()
trainX2=pd.DataFrame(trX2).sort_index()
trainY2=pd.DataFrame(trY2).sort_index()
trainX3=pd.DataFrame(trX3).sort_index()
trainY3=pd.DataFrame(trY3).sort_index()
trainX4=pd.DataFrame(trX4).sort_index()
trainY4=pd.DataFrame(trY4).sort_index()
trainX5=pd.DataFrame(trX5).sort_index()
trainY5=pd.DataFrame(trY5).sort_index()
trainX6=pd.DataFrame(trX6).sort_index()
trainY6=pd.DataFrame(trY6).sort_index()
trainX7=pd.DataFrame(trX7).sort_index()
trainY7=pd.DataFrame(trY7).sort_index()
trainX8=pd.DataFrame(trX8).sort_index()
trainY8=pd.DataFrame(trY8).sort_index()

trainXpart1=pd.concat((trainX1,trainX2),axis=0)
trainXpart2=pd.concat((trainX3,trainX4),axis=0)
trainXpart3=pd.concat((trainX5,trainX6),axis=0)
trainXpart4=pd.concat((trainX7,trainX8),axis=0)
trainXparta=pd.concat((trainXpart1,trainXpart2),axis=0)
trainXpartb=pd.concat((trainXpart3,trainXpart4),axis=0)
trainX=pd.concat((trainXparta,trainXpartb),axis=0)
trainX=trainX.values

trainYpart1=pd.concat((trainY1,trainY2),axis=0)
trainYpart2=pd.concat((trainY3,trainY4),axis=0)
trainYpart3=pd.concat((trainY5,trainY6),axis=0)
trainYpart4=pd.concat((trainY7,trainY8),axis=0)
trainYparta=pd.concat((trainYpart1,trainYpart2),axis=0)
trainYpartb=pd.concat((trainYpart3,trainYpart4),axis=0)
trainY=pd.concat((trainYparta,trainYpartb),axis=0)
trainY=trainY.values

testXpart1=pd.concat((tesX1,tesX2),axis=0)
testXpart2=pd.concat((tesX3,tesX4),axis=0)
testXpart3=pd.concat((tesX5,tesX6),axis=0)
testXpart4=pd.concat((tesX7,tesX8),axis=0)
testXparta=pd.concat((testXpart1,testXpart2),axis=0)
testXpartb=pd.concat((testXpart3,testXpart4),axis=0)
testX=pd.concat((testXparta,testXpartb),axis=0)
testX=testX.values

testYpart1=pd.concat((tesY1,tesY2),axis=0)
testYpart2=pd.concat((tesY3,tesY4),axis=0)
testYpart3=pd.concat((tesY5,tesY6),axis=0)
testYpart4=pd.concat((tesY7,tesY8),axis=0)
testYparta=pd.concat((testYpart1,testYpart2),axis=0)
testYpartb=pd.concat((testYpart3,testYpart4),axis=0)
testY=pd.concat((testYparta,testYpartb),axis=0)
testY=testY.values

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



model = Sequential()
# Model tek layerlı şekilde kurulacak.
model.add(LSTM(10, input_shape = (48605,31)))
model.add(Dense(1))
model.compile(loss='mae', optimizer = "adam")
#30 epoch yani 30 kere verisetine bakılacak.
history=model.fit(trainX, trainY, epochs=50, batch_size=20)

def predict_and_score(model, X, Y):
    # tahminleri 0-1 ile scale edilmiş halinden geri çevir
    pred = model.predict(X)
    orig_data = [Y]
    # Rmse değerlerini ölç
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)
rmse_train, train_predict = predict_and_score(model, trainX, trainY)
rmse_test, test_predict = predict_and_score(model, testX, testY)
print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)
print(history.history.keys())

#history for loss
plt.figure(figsize = (25, 2))
plt.plot(test_predict,color="red")
plt.plot(testY,color="blue")
plt.title('test result')
plt.ylabel('SOH')
plt.xlabel('veri sayısı')
plt.legend(['real', 'test_predict'], loc='upper right')
plt.show()


#loss için grafik
plt.figure(figsize = (25, 5))
plt.plot(train_predict,color="red")
plt.plot(trainY,color="blue")
plt.title('train result')
plt.ylabel('SOH')
plt.xlabel('veri sayısı')
plt.legend(['real', 'train predict'], loc='upper right')
plt.show()

#r2 score göster
from sklearn.metrics import r2_score
print(r2_score(testY, test_predict))

plt.figure(figsize = (15, 5))
plt.plot(history.history['loss'],color="red")
plt.plot(history.history['val_loss'],color="blue")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

