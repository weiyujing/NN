import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import pandas as pd
import csv
from keras.models import load_model
import matplotlib.pyplot as plt
import tushare as ts
import csv
import os
import sys

# # （1）数据读取
# df=ts.get_hist_data('600128')
#



def get_ma(filename, ma,j):
    j=int(j)

    stock = pd.read_csv(filename, parse_dates=[0])
    x_test = stock['ma' + str(ma)][j:j+4][::-1]#[30-33][29-32]。。。[1-4]

    x_test=list(x_test)

    x_test=np.array(x_test)

    normal_test = [(float(p) / x_test[0] - 1) for p in x_test]
    normal_test=np.array(normal_test)

    normal_test = np.reshape(normal_test, (1, 4, 1))

    # # （5）Building Model

    # 序列模型是一个线性层次堆栈
    model = Sequential()
    model.add(LSTM(
        input_shape=(4, 1),
        units=20,
        return_sequences=True))
    model.add(Dropout(0.2))
    # model.add(LSTM(
    # 20,
    # return_sequences=True))
    model.add(LSTM(
        20,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
        units=1))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="rmsprop")



    # # （7）predict

    model = load_model('demo_model/demo_model' + str(os.path.basename(filename)[:-4]) + str('a' + str(ma)) + '.h5')

    predicted = model.predict(normal_test)
    predicted = np.reshape(predicted, (predicted.size,))

    # 把均线数据还原
    Depredicted = (predicted + 1) *x_test[0]
    # print(len(Depredicted),Depredicted)

    return Depredicted,x_test


if __name__ == "__main__":
    #filepath = 'E:/astock_csv/new_stock_data'

    stockname = str(sys.argv[1])
    l_name = str(sys.argv[2]) #第几个模型 l_name=i
    j= str(sys.argv[3])
    filename =  str(stockname) + '.csv'
    Depredicted,x=get_ma(filename, l_name,j)

    print(Depredicted[0])




