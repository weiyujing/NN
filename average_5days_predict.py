
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

df = pro.daily(ts_code='600128.SH', start_date='20001219', end_date='20181130')
df.to_csv('E:/astock_csv/hongyegufen.csv',columns=['trade_date','open','high','low','close','ma5','ma10','ma20'])

'''

filename = 'E:/astock_csv/xingyuanhuanjing.csv'
stock=pd.read_csv(filename,parse_dates=[0])
price =list((stock['close']))
MA5=list(stock['ma5'])
MA5=MA5[4:]         #准备预处理的5日均线
Dema5=MA5[:]       #原始数据均线




# # （2）数据切分


sequence_length = 4
result = []#准备预处理
Deresult=[]#原始数据
for _ in range(len(price) - sequence_length ):
    result.append(price[_: _ + sequence_length])
    Deresult.append(price[_: _ + sequence_length])


# # （3）数据预处理

j=0
normalised_data = []

y_test_base = []
print(len(result))
for index in result:

    MA5[j] = MA5[j] / float(index[0]) - 1
    normalised_window = [(float(p) / float(index[0])- 1) for p in index]
    #MA5[j]=MA5[j]/float(index[0])- 1
    j+=1
    normalised_data.append(normalised_window)
result = normalised_data




# # （4）切分训练集 测试集
# ##   90%的数据用作训练集，剩下数据作为测试集

result = np.array(result)
Deresult = np.array(Deresult)

row = round(0.9 * result.shape[0])
train = result[:int(row), :]

x_train = train
y_train = MA5[:int(row)]
y_test_base=price[int(row):]
x_test = result[int(row):, :]  #已经预处理的测试集
Dex_test=Deresult[int(row):, :] #原始测试集

y_test = MA5[int(row):]#预处理5日线
Dema5=Dema5[int(row):] #原始5日线
clo_5th=price[-len(x_test):]#第5天真实收盘值
#print("clo5/n",len(x_test),clo_5th)

#（samples，time_step，dim） 第一个是训练样本个数，第二个是时间步，第三个就是输入维度
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
'''
xtest=[0
,-0.0068
,-0.10574
,-0.19486
]

xtest2=[0
,0.002469
,-0.04444
,-0.041975
]
xtest=np.reshape(xtest2,(1,4,1))
'''



# # （5）Building Model

#序列模型是一个线性层次堆栈
model = Sequential()
model.add(LSTM(
    input_shape=(4,1),
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
#model = load_model('demo_model.h5')

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
predicted = [] #预测预处理过的均线
predicted = model.predict(x_test)
#predicted=model.predict(xtest)
predicted = np.reshape(predicted, (predicted.size,))

#把均线数据还原
Depredicted=[]
for i in range(len(predicted)):
    Depredicted.append((predicted[i]+1)*y_test_base[i])
#print(Depredicted)
#print("x_train=\n", x_train,len(x_train) ,len(y_train),"y_train=\n", (y_train),"x_test=\n",x_test,"y_test\n",(y_test))
De5th_close=[] #由均线推第五日收盘价
for i in range(len(Depredicted)):
    sum=0
    for j in range(4):
        sum=sum+Dex_test[i][j]
    De5th_close.append(5*Depredicted[i]-sum)
#print("De\n",De5th_close)

'''

sum=0

for i in range(len(predicted)-1):
    sum+=(y_test[i]-predicted[i])
print("len=",len(predicted)-1,"sum=",sum,"avg=",sum/len(predicted))
'''



ok=0
for i in range(1,len(predicted)):
    print(De5th_close[i], clo_5th[i], clo_5th[i - 1])
    if ((De5th_close[i]-clo_5th[i-1])*(clo_5th[i]-clo_5th[i-1]))>0:

            ok = ok + 1
    if (clo_5th[i] == clo_5th[i-1]) :
        if (De5th_close[i] == De5th_close[i-1]):
            #if abs(De5th_close[i] - clo_5th[i]) < 0.1:
                ok = ok + 1
print("acc=",ok/len(predicted))
ok=0
for i in range(1,len(predicted)):

        if abs(Dema5[i]-Depredicted[i])<0.1:
               ok=ok+1
print("acc1=",ok/len(predicted))

ok=0
for i in range(1,len(predicted)):

        if abs(clo_5th[i]-De5th_close[i])<0.5:
               ok=ok+1
print("acc2=",ok/len(predicted))




# # （8）visualization

# In[65]:
plt.figure(figsize=(14,8))
plt.plot(De5th_close, label='5thclose-Prediction',color='r')

plt.plot(clo_5th, label='5thclose-true',color='b')
y = list(range(0,len(Depredicted)))

plt.legend()
plt.show()



plt.figure(figsize=(14,8))
plt.plot(Depredicted, label='ma5-Prediction',color='r')

plt.plot(Dema5, label='ma5-true',color='b')
y = list(range(0,len(predicted)))

plt.legend()
plt.show()

