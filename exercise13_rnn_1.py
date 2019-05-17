import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 设置RNN的参数
# hyperparameters超参数
lr = 0.001                  # learning_rate学习率
training_iters = 1000       # train_step上限
batch_size = 128            # batch样本数
n_inputs = 28               # MNIST data input(img shape:28x28)，输入向量大小28，因为有28列
n_steps = 28                # time steps，因为有28行
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)，有10个类

"""
rnn是如何处理？
runn将一张图片的一行pixel作为一个时间点的输入进行处理
"""

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights权重
weights = {
    # (28x28) 第一层隐含层
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),    # 矩阵[28, 128]
    # (128,10) 第二层隐含层
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))   # 矩阵[128, 10]
}

# Define bias偏置项
bias = {
    # (128,) 第一层隐含层
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),  # 生成长度为128的向量
    # (10,)  第二层隐含层
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))   # 生成[ 0.1  0.1  0.1  ...  0.1  0.1  0.1]，是一个长度为10的向量
}



# 定义RNN主体结构
def RNN(X, weights, bias):
    # 这里RNN结构是：输入层 -> LSTM层 -> 输出层,最后一个时间点的结果作为这张图片的output

    # hidden layer for input to cell
    ##########################################
    # X(128 batch, 28 steps, 28 inputs)，即128张28*28像素的image
    # X->>(128x28, 28 inputs)，原始的X是3维数据，需要把它变成2维数据才能使用weights的矩阵乘法
    X = tf.reshape(X, [-1, n_inputs])
    # X->>(128 batch * 28 steps, 128 hidden)，现在X是[128x28, 28], weights是[28, 128],结果是[(128x28),128]
    X_in = tf.matmul(X, weights['in']) + bias['in']     # (a*b)+c要满足：a的列数=b的行数=c的长度，就可以计算
    # X->>(128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])      # [128*28, 128]变成[128, 28, 128]，是为了满足LSTM输入的shape

    # LSTM cell
    ##########################################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts(c_state, m_state)，c分线state，m是主线state，分别对应单元状态ct和输出值ht
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)    # 每一步结果是一个state，这里的state全初始为0

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = _init_state, time_major = False)
    # states是(c_state, m_state)，如果是普通的rnn，则states是m_state
    # ouputs是一个list，每一步的output存在outputs中

    # hidden layer for output to cell
    ##########################################
    results = tf.matmul(states[1], weights['out'] + bias['out'])    # states[1]是m_state

    return results

# 定义cost和train_op
pred = RNN(x, weights, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)



# 训练RNN
# 定义accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)         # batch_xs是128x784的矩阵，batch_ys是128x10的矩阵
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])    # 变换形状后，batch_xs是128x28x28的数组
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})

        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
