import os
import subprocess
import sys
import datetime
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt

today=datetime.date.today()
formatted_today=today.strftime('%y%m%d')
#180117


def draw_tp_line(y_test,predicted):
    plt.figure(figsize=(14, 8))
    plt.plot(predicted, label='Prediction', color='r')
    plt.plot(y_test, label='true', color='b')
   #y = list(range(0, len(predicted)))
    plt.legend()
    plt.grid()  # 生成网格
    plt.show()

def draw_t_line(y_test,predicted):
    plt.figure(figsize=(14, 8))
    #plt.plot(predicted, label='Prediction', color='b')
    plt.plot(y_test, label='true', color='b')
    L = len(y_test)
    for i in range(L - 1):
        X = [i, i + 1]
        Y = [y_test[i], predicted[i + 1]]
        plt.plot(X, Y, color='g', marker='o', markerfacecolor='g', markersize=5)
    plt.legend()
    plt.grid()  # 生成网格
    plt.show()
if __name__ == '__main__':

    list = [ '600373', '600383', '600406', '600435' ]
    '''
    ts.set_token('8791d37f92f5bf4babf2be26716ead550a4226c65b47f1718daa0a84')
    pro = ts.pro_api()

    for l in list:
        df = ts.pro_bar(pro_api=pro, ts_code=str(l) + str('.SH'), start_date='20001219', end_date='20'+str(formatted_today),
                        ma=[i for i in range(1, 61)])
        df.to_csv( str(l) + '.csv')
    '''
    for index in list:

        stockname=index[0:6]
        predicted = []
        #print(stockname)
        #stockname=str(stockname)
        stock = pd.read_csv(stockname+str('.csv'), parse_dates=[0])
        true=stock['close'][0:30][::-1]
        #print(true)
    #运用子进程调用模型预测明日的均线数据
        for j in range(30,0,-1):#[30-33][29-32]。。。[1-4]
            emadata = []
            for i in range(1,59,3):
                #print(datetime.datetime.now())

                command = "python get_60ma.py " + str(stockname) + "   " + str(i)+ " "+ str(j)
                command2 = "python get_60ma.py " + str(stockname) + "   " + str(i+1) + " "+ str(j)
                command3 = "python get_60ma.py " + str(stockname) + "   " + str(i+2) + " "+ str(j)

                r = os.popen(command)  # 执行该命令
                r2 = os.popen(command2)  # 执行该命令
                r3 = os.popen(command3)  # 执行该命令


                info = r.readlines()  #读取命令行的输出到一个list

                for line in info:  #按行遍历
                    line = line.strip('\r\n')
                    emadata.append(str(line))


                info2 = r2.readlines()  # 读取命令行的输出到一个list
                for line in info2:  # 按行遍历
                    line = line.strip('\r\n')
                    emadata.append(str(line))


                info3 = r3.readlines()  # 读取命令行的输出到一个list
                for line in info3:  # 按行遍历
                    line = line.strip('\r\n')
                    # print(str(line))
                    emadata.append(str(line))

                #print(datetime.datetime.now())
            commands = "python get_61th.py  "+str(stockname)
            print(emadata[:5])
        #print(len(emadata))
            for j1 in emadata:
                commands=commands + "  "+str(j1)
        #print(commands)
            r1 = os.popen(commands)  # 执行该命令
            info = r1.readlines()  # 读取命令行的输出到一个list
            for line in info:  # 按行遍历
                line = line.strip('\r\n')
                print(str(stockname)+'明天的价格是：'+str(line))
                predicted.append(line)
        #print(predicted)
        predicted0 = []
        for index in range(len(predicted)):
            a = float(predicted[index][1:8])
            predicted0.append(a)
        draw_tp_line(true,predicted0)
        draw_t_line(true,predicted0)

        #print(datetime.datetime.now())