import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

filename = '600362.csv'

stock = pd.read_csv(filename, parse_dates=[0])
stock=np.array(stock)



#数据分组
result=[]
for j in range(stock.shape[1]):
    cut = []
    for i in range(61):
        cut.append(float(stock[i][j]))
    result.append(cut)

#数据预处理
normalised_data = []
y_test_base = []
for index in result:

        y_test_base.append(index[0])
        normalised_window = [(float(p) / index[0] - 1) for p in index]
        normalised_data.append(normalised_window)



row=int(0.9*stock.shape[1])#训练集
normalised_data=np.array(normalised_data)


train = normalised_data[:int(row), :]

x_train = train[:, :-1]
y_train = train[:, -1]
#row = row + 5
x_test = normalised_data[int(row):, :-1]

y_test = normalised_data[int(row):, -1]
y_test_base = y_test_base[int(row):]
y_real = stock[60][int(row):]  # 真实值

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

#全连接模型
model = Sequential()
model.add(Dense(units=10, input_dim=60))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="rmsprop")
#训练及评估模型

model.fit(x_train,
                 y_train,
                 batch_size=20,
                 epochs=100,
                  shuffle=True,)


model.save('demo_model/demo_model600362a61.h5')

# # （7）predict

model = load_model('demo_model/demo_model600362a61.h5')

predicted = model.predict(x_test)
#predicted = np.reshape(predicted, (predicted.size,))

print(predicted)
#还原
Depredicted = []
for i in range(len(predicted)):
        Depredicted.append((predicted[i] + 1) * y_test_base[i])
        print(y_test_base[i])

'''
print(Depredicted)
plt.figure(figsize=(14, 8))
plt.plot(Depredicted, label='Prediction', color='r')

plt.plot(y_real, label='true', color='b')
y = list(range(0, len(predicted)))

for i in range(len(predicted) - 1):
    X = [i, i + 1]
    Y = [y_real[i], Depredicted[i + 1]]
    plt.plot(X, Y, color='y', marker='o', markerfacecolor='g', markersize=3)

plt.legend()
plt.show()
'''

