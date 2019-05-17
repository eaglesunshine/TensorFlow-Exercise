import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 这两句专门用于下载mnist数据(如果当前文件夹内没有此数据文件)，因此它确实是涉及到网址的。而且它下载了之后会在当前目录有一个MNIST_data文件

def compute_accuracy(v_xs, v_ys):
    global prediction                                                           # 设置全局变量
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})                          # 将xs数据feed到prediction中生成预测值，重算针对test数据的输出
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))     # tf.equal()：相等返回True，等于数字1
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))          # tf.cast(x):将x的数据格式转化成dtype
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})                 # 计算测试数据集的准确率
    return result       # 输出百分比

def weight_variable(shape):     # 输入一个shape，返回一些weight_variable的参数
    inital = tf.truncated_normal(shape, stddev=0.1)      # 产生随机变量,这里是截断正态分布，stddev: 正态分布的标准差。
                                                         # 如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
    return tf.Variable(inital)                           # 通过tf.Variable构造一个variable添加进图中，Variable()构造函数
                                                         # 需要变量的初始值（是一个任意类型、任意形状的tensor），这个初始值指定variable的类型和形状。

def bias_variable(shape):       # 权重
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def conv2d(x, W):           # 定义卷积神经网络层，x是输入，W是权重
    # strides[1 x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # 返回一个二维的卷积，实现二维卷积计算.strides是步长:第一个和第第四个必须是1，第2个参数是水平跨度，第3个参数是纵向跨度。
    # padding='SAME':卷积结果跟输入的shape相同（输入边缘用0填充）；
    # padding='SAME':卷积结果比输入的shape小（输入边缘不填充）；

def max_pool_2x2(x):        # 池化层
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # None表示可以给定任意多个输入数据,每张图片28x28个像素点（黑就1，白就0），共784个像素点
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])       # 传入xs，并将xs的形状转换为[-1, 28, 28, 1],第四个参数=channel数，这里是黑白图片，所以channe是1，如果是彩色，channel数就是3
"""
这里是将一组图像矩阵x重建为新的矩阵，该新矩阵的维数为（a，28，28，1），其中-1表示a由实际情况来定。例如，x是一组图像的矩阵（假设是50张，大小为56×56），则执行
x_image = tf.reshape(x, [-1, 28, 28, 1])
可以计算a=50×56×56/28/28/1=200。即x_image的维数为（200，28，28，1）。
"""

# add ouput layer
##conv1 layer##
W_conv1 = weight_variable([5, 5, 1, 32])       # patch 5x5, in_size 1, out_size 32
# patch是image被卷积的元素块大小，in_size是image的深度，out_size是卷积结果output的深度
# 每个卷积层可以有多个filter。每个filter和原始图像进行卷积后，都可以得到一个Feature Map。因此，卷积后Feature Map的深度(个数)和卷积层的filter个数是相同的。
# 问题：这些是谁与谁连接的权重矩阵？
# 一个卷积核的所有元素与一个image卷积块中的每一个元素点乘，所有卷积核加起来的元素总数=image元素数x卷积核个数，其中卷积核个数=Feature Map的深度
# 卷积的计算：如果卷积前的图像深度为D，那么相应的filter的深度也必须为D。我们扩展一下式1，得到了深度大于1的卷积计算公式：
#                               累加f（对应元素点乘+bias）的和=1个数
b_conv1 = bias_variable([32])      # 参考卷积公式，需要的bias个数=卷积核个数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)       # relu非线性处理， output size 28x28x32，因为strides=[1,1,1,1]
h_pool1 = max_pool_2x2(h_conv1)     # 输出值，output size 14x14x32，因为strides=[1,2,2,1]，所以x、y要分别除以2

##conv2 layer##
W_conv2 = weight_variable([5, 5, 32, 64])       # patch 5x5, in_size 32, out_size 64
b_conv2 = bias_variable([64])      # 参考卷积公式，需要的bias个数=卷积核个数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)       # 算卷积，再relu非线性处理， output size 14x14x64，因为strides=[1,1,1,1]
h_pool2 = max_pool_2x2(h_conv2)     # 输出值，output size 7x7x64，因为strides=[1,2,2,1]，所以x、y要分别除以2

##func1 layer##
W_fc1 = weight_variable([7*7*64, 1024])     # 这里是平的一维数据，是一种展开
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64]->>[n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##func2 layer##
W_fc2 = weight_variable([1024, 10])     # 这里是平的一维数据，输出是10
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_erropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))     # loss这里是交叉熵（与分类的softmax对应）
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_erropy)
"""
class tf.train.AdamOptimizer
__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
相比于基础SGD算法(随机梯度下降算法)，1.不容易陷于局部优点。2.速度更快
相应参数都有默认值。
"""

sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)                    # 提取出来一部分的x和y，100个、100个地学，可以有比较快地学习速度
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))     # 对测试集的预测准确度
