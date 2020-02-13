#!/usr/bin/python
# -*- coding: <UTF-8> -*-
# By:lrh1999
# Email:lrh2201299058@outlook.com
# 本代码不可一起运行，作为TensorFlow代码示例
import tensorflow as tf

# # 创建一个常量运算，将作为一个节点加入到默认计算图中
# hello = tf.constant("hello world!")
# # 创建一个TF对话
sess = tf.Session()
# # 运行并获得结果
# print(sess.run(hello))
# # 获取shape 用.get_shape方法
# matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
# print(matrix.get_shape())
# # 不带小数点默认为int，带小数点默认为float32
# tf.reset_default_graph()
'''
异常处理
try:
except:
finally:
    sess.close()
会话经典模式2
python中的上下文管理器管理会话
with tf.Session() as sess:
    
    #使用创建好的会话来关心的结果
    print(sess.run(result))
    
'''
# 定义变量a
a = tf.Variable(1, name='a')
# 定义操作b为a+1
b = tf.add(a, 1, name='b')
# 定义操作c为b*4
c = tf.multiply(b, 4, name='c')
# 定义d为c-b
d = tf.subtract(c, b, name='d')

# logder改为自己机器上的合适路径
logdir = 'D:/log'
# 生成一个写日志的writer，并将当前的TensorFlow计算图写入日志
writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
writer.close()
'''
result = tf.add(name1,name2)
sess = tf.InteractiveSession()  设置默认会话
运行print(result.eval())
获得结果
'''

# 关闭session
sess.close()

'''
常量 tf.content
变量 
name_variable=tf.Variable(value,name)
个别变量的初始化：
init_op=name_variable.initializer()
所有变量的初始化：
init_op=tf.global_variables_initializer()
变量初始化后必须sess.run

变量赋值tf.assign(value,new_value)

'''
'''
可视化
使用tensorboard添加的4句话
tf.reset_default_graph()#清楚default graph和不断增加的节点
# logder改为自己机器上的合适路径
logdir = 'D:/log'
# 生成一个写日志的writer，并将当前的TensorFlow计算图写入日志
writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
writer.close()

用命令 进入日志目录
命令：tensorboard --logdir=/path/log 启动服务
'''
'''
tensorflow中的占位符不需要初始化操作
tf.placeholder(dtype,shape,name='tx')
feed提交数据 feed_dict={a:9,b:8}
result = sess.run(c,feed_dict={a:10.0,b:8.0})
python中可以直接将返回的两个值赋值给两个变量
'''

