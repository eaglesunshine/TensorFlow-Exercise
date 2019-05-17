import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 这两句专门用于下载mnist数据(如果当前文件夹内没有此数据文件)，因此它确实是涉及到网址的。而且它下载了之后会在当前目录有一个MNIST_data文件

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 生成随机数
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 偏置值推荐不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + bias  # 预测值, inputs是300x1的矩阵， weights是1xout的矩阵, bias是1xout的矩阵
    if activation_function is None:  # 直接输出
        outputs = Wx_plus_b
    else:  # 使用f（x）计算
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction                                                           # 设置全局变量
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})                          # 将xs数据feed到prediction中生成预测值，重算针对test数据的输出
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))     # tf.equal()：相等返回True，等于数字1
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))          # tf.cast(x):将x的数据格式转化成dtype
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})                 # 计算测试数据集的准确率
    return result       # 输出百分比

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # None表示可以给定任意多个输入数据,每张图片28x28个像素点（黑就1，白就0），共784个像素点
ys = tf.placeholder(tf.float32, [None, 10])

# add ouput layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)      # softmax就是用来分类的

# the error between prediction and real data
cross_erropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))     # loss这里是交叉熵（与分类的softmax对应）
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_erropy)

sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)                    # 提取出来一部分的x和y，100个、100个地学，可以有比较快地学习速度
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))     # 对测试集的预测准确度
