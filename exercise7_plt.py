import tensorflow as tf
import numpy as np
import matplotlib .pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 生成随机数
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 偏置值推荐不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + bias  # 预测值, inputs是300x1的矩阵， weights是1xout的矩阵, bias是1xout的矩阵
    if activation_function is None:  # 直接输出
        outputs = Wx_plus_b
    else:  # 使用f（x）计算
        outputs = activation_function(Wx_plus_b)
    return outputs

# make some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # np.newaxis分别是在行或列上增加维度，这里是在行上增加，即出来的是300行*1列的数组
noise = np.random.normal(0, 0.05, x_data.shape)  # noise噪点，没有完全按照一元二次函数的曲线走，而是在函数线的两边有很多的数据点
y_data = np.square(x_data) - 0.5 + noise

"""
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
作用为：在规定的时间内，返回固定间隔的数据。他将返回“num”个等间距的样本，在区间[`start`, `stop`]中。其中，区间的结束端点可以被排除在外。

np.random.normal(size,loc,scale): 
给出均值为loc，标准差为scale的高斯随机数（场），即正态分布。 
 参数的意义为：
    loc：float
        此概率分布的均值（对应着整个分布的中心centre）
    scale：float
        此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
    size：int or tuple of ints
        输出的shape，默认为None，只输出一个值
"""

"""
神经网络各层神经元个数的确定：
输入层：要跟输入数据的属性个数相同
输出层：要跟输出数据的属性个数相同
隐藏层：自己定
"""
xs = tf.placeholder(tf.float32, [None, 1])  # None表示可以给定任意多个输入数据
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  # 定义隐藏层l1
prediction = add_layer(l1, 10, 1, activation_function=None)  # 定义输出层

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))     # ys和prediction都是300x1的矩阵
# tf.reduce_sum(input_tensor, reduction_indices=[1])
# 则arg2 = 1，是横向对矩阵求和，原来矩阵有几行最后就得到几个值，这里得到300x1矩阵
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
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
        try:
            ax.lines.remove(lines[0])  # 先抹除线条，后面再plot线条
        except:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)      # plot是图形绘制命令，这里是绘制预测曲线,红色线条，线条宽度是5
        plt.pause(0.1)      # 暂停0.1秒


"""
tf.square(x, name=None)
对x内的所有元素进行平方操作

调用reduce_sum(arg1, arg2)时，参数arg1即为要求和的数据，arg2有两个取值分别为0和1，
通常用reduction_indices=[0]或reduction_indices=[1]来给arg2传递参数。从上图可以看出，
当arg2 = 0时，是纵向对矩阵求和，原来矩阵有几列最后就得到几个值；相似地，当arg2 = 1时，是横向对矩阵求和，原来矩阵有几行最后就得到几个值；
当省略arg2参数时，默认对矩阵所有元素进行求和，最后得到一个值。


reduce就是“对矩阵降维”的含义，下划线后面的部分就是降维的方式，在reduce_sum()中就是按照求和的方式对矩阵降维。
那么其他reduce前缀的函数也举一反三了，比如reduce_mean()就是按照某个维度求平均值，等等。

tf.reduce_mean()求平均值。
求最大值tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
参数1--input_tensor:待求值的tensor。
参数2--reduction_indices:在哪一维上求解。
参数（3）（4）可忽略
#如果不指定第二个参数，那么就在所有的元素中取平均值。


tf.reduce_sum()用于在某个维度求和。
    reduce_sum(input_tensor,axis=None,keep_dims=False, name=None,reduction_indices=None)
input_tensor:表示输入 
axis:表示在那个维度进行sum操作。 
keep_dims:表示是否保留原始数据的维度，False相当于执行完后原始数据就会少一个维度。 
reduction_indices:为了跟旧版本的兼容，现在已经不使用了。
#如果不指定第二个参数，那么计算输出所有元素的和。
"""

"""
tensorflow允许矩阵跟向量相加，产生另一个矩阵：
    C = A + B，其中C(i,j) = A(i,j) + B(j)
这种写法使我们无需在加法操作前定义一个将向量b复制到每一行而生成的矩阵。
这种隐式地复制向量b到很多位置的方式，被称为广播(broadcasting)。
"""