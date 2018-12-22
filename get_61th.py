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



def pre_tomor_close(filename, X):
    model = Sequential()
    model.add(Dense(units=10, input_dim=60))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="rmsprop")
    model = load_model('demo_model/demo_model' + str(os.path.basename(filename)[:-4]) + str('a61') + '.h5')

    normal_test = [(float(p) / X[0] - 1) for p in X]
    normal_test = np.reshape(normal_test, (1, 60))
    predicted = model.predict(normal_test)
    predicted = np.reshape(predicted, (predicted.size,))
    # print(predicted)

    # 还原

    Depredicted = (predicted + 1) * X[0]
    return Depredicted


if __name__ == "__main__":
    predata = []
    stockname = str(sys.argv[1])
    filename =   str(stockname) + '.csv'
    for i in range(2, 62):
        predata.append(float(sys.argv[i]))
    result = pre_tomor_close(filename, predata)
    print(result)





