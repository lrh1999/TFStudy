#!/usr/bin/python3
# -*- coding: <UTF-8> -*-
# By:lrh1999
# Email:lrh2201299058@outlook.com
# 《 一元线性回归》

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''
步骤：
1.生成人工数据集及其可视化
2.构建线性模型
3.定义损失函数
4.定义优化器、最小化损失函数
5.训练结果的可视化
6.利用学习到的模型进行预测
'''
# 设置随机数种子
np.random.seed(5)  # 产生相同的随机数序列
# 直接采用numpy生成等差数列的方法，生成100个点，每个点的取值在-1~1之间
x_data = np.linspace(-1, 1, 100)
# 设置y=2x+1的噪声，其中噪声的维度和x_data一致，
# 其中np.random.randn函数是从标准正态分布中返回一个或多个样本值  N(0,1) 0为均值，1为标准差
# 实参的前面加上*和**时，就意味着拆包。单个*表示将元组拆成一个个单独的实参
y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4
# 画出散点图
plt.scatter(x_data, y_data)
# 画出原线性函数
plt.plot(x_data, 2 * x_data + 1.0, color='red', linewidth=3)
plt.show()

# 定义训练数据的占位符，x是特征值，y是标签值
x = tf.placeholder('float', name='x')
y = tf.placeholder('float', name='y')


# 定义模型函数
def model(x, w, b):
    return tf.multiply(x, w) + b


'''
 创建变量，
· 变量声明函数：tf.Variable
·变量的初始值可以是随机数，常数，或是通过其他变量的初始值计算得到
'''
# 构建线性函数的斜率和截距，变量w,b
w = tf.Variable(1.0, name='w0')
b = tf.Variable(0.0, name='b0')
# pred是预测值，前向计算
pred = model(x, w, b)
# 设置迭代次数
train_epochs = 10
# 学习率
learning_rate = 0.05
# 控制显示loss值的粒度
display_step = 10
'''
定义损失函数
·损失函数用于描述预测值与真实值之间的误差，从而指导模型收敛方向
·常见损失函数：均方差和交叉熵
'''
# 采用均方差作为损失函数
loss_function = tf.reduce_mean(tf.square(y - pred))

'''
定义优化器
·定义优化器Optimizer,初始化一个GradientDescentOptimizer
·设置学习率和优化目标：最小化损失
'''
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 声明会话
sess = tf.Session()
'''
变量初始化
·在真正执行计算前，需要将所有变量初始化
·通过tf.global_variable_initializer()函数可以实现对所有变量的初始化
'''
init = tf.global_variables_initializer()  # 初始化节点要先run一下
sess.run(init)

'''
模型训练阶段，，设置迭代轮次，每次通过将样本逐个输入模型，
进行梯度下降优化操作，每轮迭代后，绘出模型曲线
'''
'''
显示损失值
'''
# 开始训练，轮数为 epoch,采用SGD随机梯度下降优化方法
# zip()  组装函数将x_data和y_data组装起来返回两个数
step = 0  # 记录训练步数
loss_list = []  # 用于保存loss值的列表

for epoch in range(train_epochs):
    for xs, ys in zip(x_data, y_data):
        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})

        # 显示损失值 loss
        # display_step:控制报告的粒度
        # 例如，如果 display_step 设为2，则将每训练2个样本输出一次损失值
        # 与超参数不同，修改 display_step 不会更改模型所学习的规律
        loss_list.append(loss)
        step = step + 1
        if step % display_step == 0:
            print('Train Epoch:', '%02d' % (epoch + 1), 'Step:%03d' % step, 'loss=', "{:.9f}".format(loss))

    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    plt.plot(x_data, w0temp * x_data + b0temp)  # 画图
plt.plot(loss_list)
plt.show()

'''
利用模型进行预测
'''
x_text = 3.21
predict = sess.run(pred, feed_dict={x: x_text})
print('预测值：%f' % predict)

target = 2 * x_text + 1.0
print('目标值：%f' % target)

sess.close()
