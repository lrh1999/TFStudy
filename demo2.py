#!/usr/bin/python3
# -*- coding: <UTF-8> -*-
# By:lrh1999
# Email:lrh2201299058@outlook.com
"""
《波士顿房价预测-多元线性回归》
使用TensorFlow进行算法设计与训练的核心步骤
1.准备数据(由于特征数据取值范围不同，对结果的影响各不相同，所以要进行数据归一化)
2.构建模型
3.训练模型
4.进行预测

所有带标签的样本都要参与训练吗？
不应该，需要做一个划分，划分为训练集与测试集，有必要时还可以划分出一部分验证集
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# 读取数据文件
df = pd.read_csv("data/boston.csv", header=0)  # 指明要不要把头放进来 0为不放
print(df.describe())  # 打印数据摘要

# 获取df值
df = df.values
# 把df 转换为 np 的数组格式
df = np.array(df)

'''
不同取值范围的特征数据有适合自己的学习率。当尺度相差太大的多个特征数据一起训练时，学习率如果取匹配小尺度特征数据的值，则对大尺度特征数据来说收敛太慢；学习率如果取匹配大尺度特征数据的值，则对小尺度特征数据来说就没法收敛。归一化后，所有特征数据都在一个尺度上，就容易找到合适的学习率对所有的特征数据都能收敛也不太慢。归一化只在多个特征数据时有意义，只有一个特征数据时，修改学习率即可，没有必要归一化。同理，标签值只有一个，归一化没有意义。

另外，我们的目标是预测标签值。当特征数据归一化后，特征数据的值变化，计算出的对应的w值也会变，不影响预测值。而如果标签值归一化了，最后预测出的值也变化了，最后还得安原来归一化的比例通过乘除原样还原回去。

总结来说，原因有三：

1、归一化是为了在多个不同尺度特征数据时找到合适的学习率，标签值只有一个，归一化没有意义；

2、归一化是为了能为特征数据找到合适的学习率，使得训练过程能收敛并且有一定的速度，而标签值和学习率没有直接关系；

3、特征参数归一化后，在预测时不需要还原，而标签值如果归一化了，预测时还得还原回去，两次计算没有意义。
'''
# 对特征数据【0到11】列 做（0-1）归一化
for i in range(12):
    df[:, i] = (df[:, i] - df[:, i].min()) / (df[:, i].max() - df[:, i].min())
# 列表切片操作详见：https://www.jianshu.com/p/15715d6f4dad
# x_data 为归一化后的前12列特征数据
x_data = df[:, :12]
# y_date 为最后一列标签数据 标签不做归一化
y_data = df[:, 12]

# 定义特征数据和标签数据的占位符   [None,12]表示多少行不知道不在乎，但是有12列
x = tf.placeholder(tf.float32, [None, 12], name='X')
y = tf.placeholder(tf.float32, [None, 1], name='Y')

# 定义模型函数 定义一个命名空间
with tf.name_scope("Model"):
    # w 初始化值为shape=(12,1)的随机数  标准差0.01
    w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name="W")

    # 初始化 b 值为1.0
    b = tf.Variable(1.0, name='b')

    # w和x是矩阵相乘，用matmul，不能用mutiply或者*
    def model(x, w, b):
        return tf.matmul(x, w) + b

    # 预测计算操作，前向计算节点
    pred = model(x, w, b)

# 模型训练 设置超参数
# 迭代轮次
train_epochs = 50
# 学习率
learning_rate = 0.01

# 定义损失函数（均方差）
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y - pred, 2))
# 创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
# 声明会话
sess = tf.Session()
# 定义初始化变量的操作
init = tf.global_variables_initializer()

# TensorBoard可视化
# 设置日志目录
logdir = 'd:/log'

# 创建一个操作，用于记录损失值loss，后面在TensorBoard中的SCALABS栏可见
sum_loss_op = tf.summary.scalar("loss",loss_function)

# 把所有的需要记录的摘要日志文件的合并，方便一次性写入
merged = tf.summary.merge_all()


# 启动会话
sess.run(init)

# 创建摘要writer,将计算图写入摘要文件，后面在TensorBoard中的SCALABS栏可见
writer = tf.summary.FileWriter(logdir, sess.graph)   # sess.graph表示计算图

loss_list = []  # 用于保存loss值
# 模型训练 迭代训练
for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs, ys in zip(x_data, y_data):
        xs = xs.reshape(1, 12)  # Feed数据必须和Placeholder的shape一致
        ys = ys.reshape(1, 1)

        _, summary_str, loss = sess.run([optimizer,sum_loss_op, loss_function], feed_dict={x: xs, y: ys})

        writer.add_summary(summary_str, epoch)
        loss_sum = loss_sum + loss
    # 打乱数据顺序
    xvalues, yvalues = shuffle(x_data, y_data)

    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    loss_average = loss_sum / len(y_data)
    loss_list.append(loss_average)    # 每轮添加一次
    print("epoch=", epoch + 1, "loss=", loss_average, "b=", b0temp, "/nw=", w0temp)

plt.plot(loss_list)
plt.show()
n = np.random.randint(506)  # 随机确定一条数据看看效果
print("第%d条" % n)
x_test = x_data[n].reshape(1, 12)
predict = sess.run(pred, feed_dict={x: x_test})
print("预测值：%f" % predict)
print("标签值：%f" % y_data[n])
