
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

# # （1）数据读取
#df=ts.get_hist_data('600128')
#
'''
ts.set_token('8791d37f92f5bf4babf2be26716ead550a4226c65b47f1718daa0a84')
pro = ts.pro_api()

df = pro.daily(ts_code='600128.SH', start_date='20001219', end_date='20181114')
df.to_csv('E:/astock_csv/hongyegufen.csv',columns=['trade_date','open','high','low','close'])
'''


filename = 'E:/astock_csv/hongyegufen.csv'
stock=pd.read_csv(filename,parse_dates=[0])
price =list((stock['close']))


# # （2）数据切分
# ![image.png](attachment:image.png)

sequence_length = 30
result = []
for _ in range(len(price) - sequence_length + 1):
    result.append(price[_: _ + sequence_length])


# # （3）数据预处理

normalised_data = []
for index in result:
    normalised_window = [(float(p) / float(index[0])- 1) for p in index]
    normalised_data.append(normalised_window)
result = normalised_data



# # （4）切分训练集 测试集
# ##   90%的数据用作训练集，剩下数据作为测试集

result = np.array(result)

row = round(0.9 * result.shape[0])
train = result[:int(row), :]
x_train = train[:, :-1]
y_train = train[:, -1]
x_test = result[int(row):, :-1]
y_test = result[int(row):, -1]
#（samples，time_step，dim） 第一个是训练样本个数，第二个是时间步，第三个就是输入维度
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
print(x_train)
'''
x_train=result[:,:-1]
y_train=result[:,-1]
x_test=[6.23,6.24,6.16,6.17,5.9,5.84,5.88,5.3,5.07,5.01,4.93,5,4.82,4.93,5.29,5.28,5.28,5.27,5.32,5.26
,5.35,5.5,5.52,6.07,6.68,7.35,8.09,8.9,9.79]
x_test=np.array(x_test)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (1, 29, 1))
y_test=10.77
'''

# # （5）Building Model

#序列模型是一个线性层次堆栈
model = Sequential()
model.add(LSTM(
    input_shape=(29,1),
    units=20,
    return_sequences=True))
model.add(Dropout(0.2))
#model.add(LSTM(
   # 20,
    #return_sequences=True))
model.add(LSTM(
    20,
    return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(
    units=1))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="rmsprop")

# # （6）Train Model

model.fit(
	    x_train,
	    y_train,
	    batch_size=100,
	    epochs =100,
        shuffle = True,    
)

model.save('demo_model.h5') 


# # （7）predict



#model = load_model('demo_model.h5')
predicted = []
predicted = model.predict(x_test)
predicted = np.reshape(predicted, (predicted.size,))
print(predicted)
sum=0
'''
for i in range(len(predicted)-1):
    sum+=(y_test[i]-predicted[i])
print("len=",len(predicted)-1,"sum=",sum,"avg=",sum/len(predicted))
'''
# # （8）visualization

# In[65]:



plt.figure(figsize=(14,8))
plt.plot(predicted, label='Prediction',color='r')

plt.plot(y_test, label='true',color='b')
y = list(range(0,len(predicted)))

plt.legend()
plt.show()


# In[66]:



plt.figure(figsize=(15,8.5))

y = list(range(0,len(predicted)))
plt.scatter(y,predicted,label='Prediction',color='r')

plt.scatter(y,y_test, label='true',color='b')
plt.legend()

