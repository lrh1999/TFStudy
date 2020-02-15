#!/usr/bin/python3
# -*- coding: <UTF-8> -*-
# By:lrh1999
# Email:lrh2201299058@outlook.com
"""
手写数字识别 多层神经网络
"""
import tensorflow as tf
from time import time
# 导入TensorFlow提供的数据读取模块
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 读取MNIST数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义占位符
x = tf.placeholder(tf.float32, [None, 784], name='X')
y = tf.placeholder(tf.float32, [None, 10], name='Y')

# 构建隐含层神经元数量
H1_NN = 256

W1 = tf.Variable(tf.random_normal([784, H1_NN]))
b1 = tf.Variable(tf.zeros([H1_NN]))
# 设置 relu 激活函数
Y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# 构建输出层
W2 = tf.Variable(tf.random_normal([H1_NN, 10]))
b2 = tf.Variable(tf.zeros([10]))

forward = tf.matmul(Y1, W2) + b2
pred = tf.nn.softmax(forward)  # 生成多分类预测结果
# 交叉熵
loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),
                                              reduction_indices=1))

# 设置训练参数
train_epochs = 100   # 轮
batch_size = 50
total_batch = int(mnist.train.numexamples/batch_size)
display_step = 1  # 控制显示粒度
learning_rate = 0.01

# 选择优化器
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss_function)
# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 记录训练开始时间
start_time = time()

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size)  # 读取批次数据
        sess.run(optimizer,feed_dict={x: xs, y: ys})  # 执行批次训练

    # total_batch 个批次训练完成后，使用验证数据计算误差与准确率
    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x: mnist.validation.images,
                                    y: mnist.validation.labels})
    if (epoch+1) % display_step == 0:
        print("Train Epoch:", '%02d' % (epoch+1),
              "Loss=", "{:.9f}".format(loss),"Accuracy=", "{:.4f}".format(acc))

# 显示运行总时间
duration = time()-start_time
print("Train Finished takes:","{:.2f}".format(duration))
