#!/usr/bin/python3
# -*- coding: <UTF-8> -*-
# By:lrh1999
# Email:lrh2201299058@outlook.com

"""
MNIST手写手写数字识别单神经元
测试集需要满足两个条件：
1.规模足够大，可产生具有统计意义的结果
2.能代表整个数据集，特征应当与训练集相同
逻辑回归 2分类可用对数损失函数 多分类 交叉熵损失函数
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

'''
独热编码（one_hot）：
分类任务一般用独热编码
一中稀疏向量，其中：
一个元素设为1
所有其他元素为0
将离散型数据特征使用one-hot编码，会让特征之间的距离计算更加合理
'''


#  定义可视化函数
def plot_images_labels_prediction(images,
                                  label,
                                  prediction,
                                  index,
                                  num=10
                                  ):
    fig = plt.gcf()  # 获取当前图表，Get Current Figure
    fig.set_size_inches(10, 12)  # 1英寸等于2.54cm
    if num > 25:
        num = 25  # 最多显示25个子图
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)  # 获取当前要处理的子图
        ax.imshow(np.reshape(images[index], (28, 28)),  # 显示第index个图像
                  cmap='binary')

        title = "label=" + str(np.argmax(label[index]))  # 构建该图上要显示的tittle
        if len(prediction)>0:
            title += ",predict=" + str(prediction[index])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])  # 不显示坐标轴
        ax.set_yticks([])
        index += 1
    plt.show()


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('train images shape:', mnist.train.labels.shape, 'lables shape:',
      mnist.train.labels.shape)
# len(mnist.train.images[0])
# # mnist.train.images[0].shape
# # mnist.train.images[0]
# mnist.train.images[0].reshape(28, 28)
#
#
# # 可视化 image
# def plot_image(image):
#     # 图像的模式参数 cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。
#     plt.imshow(image.reshape(28, 28), cmap='binary')
#     plt.show()
#
#
# # 显示图像第20000张图
# plot_image(mnist.train.images[20000])
# # argmax显示数组最大值的位置
# print(np.argmax(mnist.train.labels[1]))
#
# # 取10条数据用mnist包里面的方法，附带打乱数据 在执行一次的话就取没取过的10条
# batch_images_xs, batch_labels_ys = mnist.train.next_batch(batch_size=10)
# print(batch_images_xs.shape,batch_labels_ys.shape)

# 定义待输入数据的占位符
# mnist 中的图片共有 28*28=784个像素点
x = tf.placeholder(tf.float32, [None, 784], name='X')

# 0-9 共10个数字类别
y = tf.placeholder(tf.float32, [None, 10], name='Y')

# 定义模型变量
# 以正态分布的随机数初始化权重W，以常数0初始化偏置b
W = tf.Variable(tf.random_normal([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

# 定义前向计算
forward = tf.matmul(x, W) + b

# softmax结果概率分类
pred = tf.nn.softmax(forward)

# 定义损失函数,交叉熵  reduction_indices=1取均值
loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# 设置训练参数
train_epochs = 150  # 训练轮数
batch_size = 50  # 单次训练样本大小
total_batch = int(mnist.train.num_examples / batch_size)  # 一轮训练多少次
display_step = 1  # 显示粒度
learning_rate = 0.01  # 学习率

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 定义准确率 检测预测类别tf.argmax(pred,1)与实际类别tf.argmax(y,1)的匹配情况  其中的1指列维度 而0为行维度 找最大值
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# 准确率，将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()  # 声明会话
init = tf.global_variables_initializer()  # 变量初始化
sess.run(init)

# 模型训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size)  # 读取批次数据
        sess.run(optimizer, feed_dict={x: xs, y: ys})  # 执行批次训练
    # total_batch个批次训练完成后，使用验证数据计算误差与准确率：验证集没有分批
    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
    # 打印训练过程中的详细信息
    if (epoch + 1) % display_step == 0:
        print("Train Epoch:", '%02d' % (epoch + 1), "Loss=", "{:.9f}".format(loss), "Accuracy=", "{:.4f}".format(acc))

print("Train Finished!")

# 用测试数据集测试
accu_test = sess.run(accuracy,
                     feed_dict={x: mnist.test.images, y: mnist.test.labels})
print('Test Accuracy:', accu_test)

# 应用模型进行预测 由于pred预测结果是one-hot编码格式，所以需要转换为0~9数字
prediction_result = sess.run(tf.argmax(pred, 1),
                             feed_dict={x: mnist.test.images})
# 查看预测结果中的前十项
print(prediction_result[0:10])
