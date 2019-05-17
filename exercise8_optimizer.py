import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt

def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  # 生成随机数
            tf.summary.histogram(layer_name+'/weights', Weights)        # 查看变化的变量
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  # 偏置值推荐不为0
            tf.summary.histogram(layer_name + '/bias', bias)             # 第一个参数是name，第二个参数是values（实数张量）
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + bias  # 预测值, inputs是300x1的矩阵， weights是1xout的矩阵, bias是1xout的矩阵
        if activation_function is None:  # 直接输出
            outputs = Wx_plus_b
        else:  # 使用f（x）计算
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

# make some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # np.newaxis分别是在行或列上增加维度，这里是在行上增加，即出来的是300行*1列的数组
noise = np.random.normal(0, 0.05, x_data.shape)  # noise噪点，没有完全按照一元二次函数的曲线走，而是在函数线的两边有很多的数据点
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # None表示可以给定任意多个输入数据
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)  # 定义隐藏层l1
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)  # 定义输出层

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]), name="loss")     # ys和prediction都是300x1的矩阵
    tf.summary.scalar('loss', loss)
# tf.reduce_sum(input_tensor, reduction_indices=[1])
# 则arg2 = 1，是横向对矩阵求和，原来矩阵有几行最后就得到几个值，这里得到300x1矩阵
with tf.name_scope('itrain'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()     # 合并在一起
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(init)
fig = plt.figure()                 # 生成一个图片框
ax = fig.add_subplot(1, 1, 1)      # 连续性画图
# fig.add_subplot(xyi)等同于plt.subplot(xyi)#将画布分成x*y的块，这个图在第i个块上显示
ax.scatter(x_data, y_data)         # 点的形式plt上来，scatter是将图ax绘制成散点图，括号内是输入x和y
plt.ion()                          # ion()使得show（）之后 不暂停，可以连续画图
plt.show()                         # 显示真实数据
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # 使用feed_dict的好处是可以采用多轮小批量数据训练，而不是一次就all in
    if i % 50 == 0:
        #print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
