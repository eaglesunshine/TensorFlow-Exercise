import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data                         # 从0到9数字的data
y = digits.target
y = LabelBinarizer().fit_transform(y)   # labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3)

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 生成随机数
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 偏置值推荐不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + bias  # 预测值, inputs是300x1的矩阵， weights是1xout的矩阵, bias是1xout的矩阵
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:  # 直接输出
        outputs = Wx_plus_b
    else:  # 使用f（x）计算
        outputs = activation_function(Wx_plus_b)
    # tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

# define placeholder for inputs to network
keep_prob  = tf.placeholder(tf.float32)         #保持多少的结果不被dropout,参数表示百分比
xs = tf.placeholder(tf.float32, [None, 64])     # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add ouput layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)      # softmax就是用来分类的

# the error between prediction and real data
cross_erropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))     # loss这里是交叉熵（与分类的softmax对应）
tf.summary.scalar('loss', cross_erropy)     # 用来显示标量信息
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_erropy)

sess = tf.Session()
merged = tf.summary.merge_all()
# merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。

# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)      # 指定一个文件保存图
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

# important step
sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})    # 500次训练
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})     # 计算train的ys的loss点
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})        # 计算test的ys的loss的点
        train_writer.add_summary(train_result, i)       # train_result是summary类型的，需要放入train_writer中，i步数（x轴）
        test_writer.add_summary(test_result, i)

"""
train_step每次按keep_prob保留率进行训练，计算得出一次训练得到的prediction是一个输入样本数x标签数的矩阵，分别对应每个样本的对应可能的标签的概率值。
loss采用交叉熵计算，即计算每个ys*log(prediction)并求和，再取负数，再求平均数就是loss。
=>计算loss需要使用两个参数，一个ys矩阵，由summary函数给定了，比如上面的y_train和y_test矩阵；另一个prediction矩阵就是当前训练出来的ouputs矩阵，
它在一轮循环里面是不变的。
=>train_writer图里保存到的一系列每隔50次训练取的一次训练的loss值，横坐标为i（训练次数），纵坐标为loss值。这个loss值表达的是输入X_train计算的预测值跟训练集真实值的误差。
  test_writer图里保存到的一系列每隔50次训练取的一次训练的loss值，横坐标为i（训练次数），纵坐标为loss值。这个loss值表达的是这一次X_test计算的预测值跟测试集真实值的误差。？？？
  
问题是：走到test_result时，train_step是否会使用xs: X_test, ys: y_test训练一遍得出新的prediction矩阵，还是沿用老prediction矩阵。

一次训练的流程：
传入xs和ys，调用train_step;
由train_step采用梯度下降法训练权重矩阵，何时停止输出？
一次前向输出+一次反向误差传播，就截止，输出outputs是由前向计算得到的结果。

tensorflow运作模式
        1.tensorflow是用python先构建一个图，然后通过外部运算优化得到结果
        2.向模型不断喂入数据，然后给出要不断优化的对象loss，根据loss的走势不断优化模型得到结果
一旦启动sess.run(),就是启动数据流图，就把程序看出一张数据流图，看所要节点的输入输出，该节点的输入的依赖节点，看数据源到该节点数据是怎么流的。
"""