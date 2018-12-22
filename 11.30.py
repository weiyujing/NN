
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

# # （1）数据读取
#df=ts.get_hist_data('600128')
#

def gci(filepath):
    # 遍历filepath下所有文件，包括子目录

    filename=[]
    file=[]
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)

        filename.append(os.path.basename(fi_d)[:-4])  # 返回文件名
        file.append(fi_d)
    return file,filename


def get_ma(filename,ma):
    stock = pd.read_csv(filename, parse_dates=[0])
    price = list((stock['ma' + str(ma)])[59:])

    # # （2）数据切分

    sequence_length = 5
    result = []
    Deresult = []#原始数据
    for _ in range(len(price) - sequence_length + 1):
        result.append(price[_: _ + sequence_length])
        Deresult.append(price[_: _ + sequence_length])

    # # （3）数据预处理

    normalised_data = []
    y_test_base = []
    for index in result:

        y_test_base.append(index[0])
        normalised_window = [(float(p) / index[0] - 1) for p in index]
        normalised_data.append(normalised_window)

    result = normalised_data

    # # （4）切分训练集 测试集
    # ##   90%的数据用作训练集，剩下数据作为测试集

    result = np.array(result)
    Deresult = np.array(Deresult)
    row = round(0.9 * result.shape[0])
    #print("row\n",row)
    train = result[:int(row), :]

    x_train = train[:, :-1]
    y_train = train[:, -1]
    row=row+5
    x_test = result[int(row):, :-1]

    y_test = result[int(row):, -1]
    y_test_base=y_test_base[int(row):]
    y_real=price[int(row)+4:]#未压缩的真实值
    #print("zhenshi\n",len(y_real),y_real)
    # （samples，time_step，dim） 第一个是训练样本个数，第二个是时间步，第三个就是输入维度
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

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

    # # （6）Train Model
'''
    model.fit(
        x_train,
        y_train,
        batch_size=100,
        epochs=100,
        shuffle=True,
    )

    model.save('demo_model/demo_model'+str(os.path.basename(filename)[:-4])+str('a'+str(ma))+'.h5')
'''


    model = load_model('demo_model/demo_model'+str(os.path.basename(filename)[:-4])+str('a'+str(ma))+'.h5')
    predicted = []
    predicted = model.predict(x_test)
    predicted = np.reshape(predicted, (predicted.size,))


    # 把均线数据还原
    Depredicted = []
    for i in range(len(predicted)):
        Depredicted.append((predicted[i] + 1) * y_test_base[i])
    #print(len(Depredicted),Depredicted)
    if(int(ma)==1):
        csvFile = open(str('60_pre_out/'+os.path.basename(filename)[:-4]) + ".csv", "w")
    else:
        csvFile = open(str('60_pre_out/'+os.path.basename(filename)[:-4])+".csv", "a")
    writer = csv.writer(csvFile)

    # 分批写入
    writer.writerow(Depredicted)
    # 一次写入
   # writer.writerows([fileHeader, d1, d2])

    csvFile.close()
    return len(predicted)


if __name__ == "__main__":
    filename = 'train_data/600362.csv'
    lens=0
    for i in range(1,61):
    
        lens=get_ma(filename,i)

    stock = pd.read_csv(filename, parse_dates=[0])
    stock = stock['close'][-lens:]
    csvFile = open(str('60_pre_out/'+os.path.basename(filename)[:-4])+".csv", "a")
    writer = csv.writer(csvFile)
    writer.writerow(stock)
    csvFile.close()





