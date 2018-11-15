#基于TensorFlow，搭建一个RNN，教会神经网络进行二进制加法


import tensorflow as tf
import numpy as np


# 一个字典，隐射一个数字到其二进制的表示
# 例如 int2binary[3] = [0,0,0,0,0,0,1,1]
int2binary = {}

# 最多8位二进制
binary_dim = 8

# 在8位情况下，最大数为2^8 = 256
largest_number = pow(2,binary_dim)

# 将[0,256)所有数表示成二进制
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)

# 建立字典
for i in range(largest_number):
    int2binary[i] = binary[i]

def binary_generation(numbers, reverse = False):#返回numbers中所有数的二进制表达，
    '''
    例如 numbers = [3, 2, 1]
    返回 ：[[0,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]'

    如果 reverse = True, 二进制表达式前后颠倒，
    这么做是为训练方便，因为训练的输入顺序是从低位开始的

    numbers : 一组数字
    reverse : 是否将其二进制表示进行前后翻转
    '''
    binary_x = np.array([ int2binary[num] for num in numbers], dtype=np.uint8)

    if reverse:
        binary_x = np.fliplr(binary_x)

    return binary_x

def batch_generation(batch_size, largest_number):
    '''
    生成batch_size大小的数据，用于训练或者验证

    batch_x 大小为[batch_size, biniary_dim, 2]
    batch_y 大小为[batch_size, biniray_dim]
    '''

    # 随机生成batch_size个数
    n1 = np.random.randint(0, largest_number//2, batch_size) #//向下取整 225//2=112
    n2 = np.random.randint(0, largest_number//2, batch_size)
    #n1= np.array([109,34])
    #n2 = np.array([95,64])
    #n1 = np.array([109])
    #n2 = np.array([95])
    # 计算加法结果
    add = n1 + n2
    #print(add)

    # int to binary
    binary_n1 = binary_generation(n1, True)  #batch_size*8 1个整数由8位表示
    binary_n2 = binary_generation(n2, True)
    '''
      n1=[109  34  84]
      n2 = [95 64 50]
      binary_n1 = [[1 0 1 1 0 1 1 0]
                  [0 1 0 0 0 1 0 0]
                  [0 0 1 0 1 0 1 0]]
     '''
    batch_y = binary_generation(add, True)

    # 堆叠，因为网络的输入是2个二进制
    '''
    a=[1,2,3,4]
b=[5,6,7,8]
c=[9,10,11,12]

print("增加一维，新维度的下标为0")
d=np.stack((a,b,c),axis=0)
print(d)

print("增加一维，新维度的下标为1")
d=np.stack((a,b,c),axis=1)

增加一维，新维度的下标为0
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
增加一维，新维度的下标为1
[[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]

    '''
    batch_x = np.dstack((binary_n1, binary_n2))#[[n1],[n2]]
    '''
    [[[1 1]
  [0 1]
  [1 1]
  [1 1]
  [0 1]
  [1 0]
  [1 1]
  [0 0]]

 [[0 0]
  [1 0]
  [0 0]
  [0 0]
  [0 0]
  [1 0]
  [0 1]
  [0 0]]

 [[0 0]
  [0 1]
  [1 0]
  [0 0]
  [1 1]
  [0 1]
  [1 0]
  [0 0]]]
    '''

    return batch_x, batch_y, n1, n2, add

def binary2int(binary_array):
    '''
    将一个二进制数组转为整数
      enumerate()
    函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在
    for 循环当中。
    '''
    out = 0

    for index, x in enumerate(reversed(binary_array)):
        out += x*pow(2, index)
    return out
batch_size = 2
# LSTM的个数，就是隐层中神经元的数量
lstm_size = 10
# 隐层的层数
lstm_layers =3

# 输入，[None, binary_dim, 2],
# None表示batch_size, binary_dim表示输入序列的长度=8，2表示每个时刻有两个输入
x = tf.placeholder(tf.float32, [None, binary_dim, 2], name='input_x')

# 输出
y_ = tf.placeholder(tf.float32, [None, binary_dim], name='input_y')
# dropout 参数
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# 搭建LSTM层（看成隐层）
# 有lstm_size个单元
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# dropout
drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
# 一层不够，就多来几层
def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)
cell = tf.contrib.rnn.MultiRNNCell([ lstm_cell() for _ in range(lstm_layers)])#两层

# 初始状态，可以理解为初始记忆
initial_state = cell.zero_state(batch_size, tf.float32)

# 进行forward，得到隐层的输出
# outputs 大小为[batch_size,binary_dim, lstm_size]  LSTMStateTuple  shape=(2, 10)
output1, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)
# output1=shape=(2, 8, 10) [batchsize,binary,lstmsize]


# 建立输出层
weights = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.01))#1代表输出一个和向量。这个函数产生正太分布，均值和标准差自己设定。
bias = tf.zeros([1])

# [batch_size, lstm_size,binary_dim] ==> [batch_size*binary_dim, lstm_size]???
outputs = tf.reshape(output1, [-1, lstm_size])
# 得到输出, logits大小为[batch_size*binary_dim, 1]
logits = tf.sigmoid(tf.matmul(outputs, weights))
# [batch_size*binary_dim, 1] ==> [batch_size, binary_dim]
predictions = tf.reshape(logits, [-1, binary_dim])

#损失函数和优化方法

cost = tf.losses.mean_squared_error(y_, predictions)
optimizer = tf.train.AdamOptimizer().minimize(cost)


#训练
'''[[[1 1]    
 [0 1]
 [1 1]
 [1 1]
 [0 1]
 [1 0]
 [1 1]
 [0 0]]

[[0 0]
 [1 0]
 [0 0]
 [0 0]
 [0 0]
 [1 0]
 [0 1]
 [0 0]]

[[0 0]
 [0 1]
 [1 0]
 [0 0]
 [1 1]
 [0 1]
 [1 0]
 [0 0]]]
 input_y:
 [[0 0 1 1 0 0 1 1]
[0 1 0 0 0 1 1 0]
[0 1 1 0 0 0 0 1]]
 '''
steps = 200000
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    iteration = 1
    for i in range(steps):
        # 获取训练数据
        input_x, input_y,_,_,_ = batch_generation(batch_size, largest_number)

        _, loss = sess.run([optimizer, cost], feed_dict={x:input_x, y_:input_y, keep_prob:0.5})
        out,out1=sess.run([output1,outputs],feed_dict={x:input_x, y_:input_y, keep_prob:0.5})
        #print("out1=",output1,"\nouts=",outputs)
        #print("out1=", out, "\nouts=", out1)
        if iteration % 1000 == 0:
            print('Iter:{}, Loss:{}'.format(iteration, loss))
        iteration += 1

    # 训练结束，进行测试
    val_x, val_y, n1, n2, add = batch_generation(batch_size, largest_number)
    result = sess.run(predictions, feed_dict={x:val_x, y_:val_y, keep_prob:1.0})

    # 左右翻转二进制数组。因为输出的结果是低位在前，而正常的表达是高位在前，因此进行翻转
    result = np.fliplr(np.round(result))
    result = result.astype(np.int32)

    for  b_x, b_p, a, b, add in zip(np.fliplr(val_x), result, n1, n2, add):
        print('{}:{}'.format(b_x[:,0], a))
        print('{}:{}'.format(b_x[:,1], b))
        print('{}:{}\n'.format(b_p, binary2int(b_p)))

