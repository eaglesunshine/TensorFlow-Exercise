import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt     # 可视化模块

# RNN预测函数波形，从sin(x)预测cos(x)
# 本质是一个回归问题的预测，从一个sequence预测另一个sequence

# 设置 RNN 的参数
BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006

# 定义一个生成数据的 get_batch function:
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]
    # newaxis顾名思义就是插入新维度的意思，所以seq和res变三维，而xs还是二维

# 定义 LSTMRNN 的主体结构
class LSTMRNN(object):
    # 第一步定义 class 中的 __init__ 传入各种参数:
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        # 给变量赋值
        self.n_steps = n_steps          # 一个RNN inputs结构有几步，步数=RNN Cell数目
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        # 定义placeholder
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')    # xs是一个三维数组
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        # 创建输入层
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        # 创建隐含层
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        # 创建输出层
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        # 定义代价函数
        with tf.name_scope('cost'):
            self.compute_cost()
        # 定义训练函数
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)  # 优化每一次的平均损失
            """
            tf.train.AdamOptimizer():
            此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
            相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
            """

    # 设置 add_input_layer 功能, 添加 input_layer:
    def add_input_layer(self,):     #第一层
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # 三维转二维才能输入：->(batch*n_step, in_size)
        # 获取权重Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # 获取偏置项bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # 计算第一层的输出l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # 二维变三维，因为placeholder的输入xs是三维：reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    # 设置 add_cell 功能, 添加 cell, 注意这里的 self.cell_init_state, 因为我们在 training 的时候, 这个地方要特别说明.
    def add_cell(self):       # 第二层
        # cell_size表示Cell中基本神经单元的个数， forget_bias是遗忘门的偏置项(bias设为1时会更健壮)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        # LSTM中每一步都有一个state，现在要设置初始的state，这里设为0
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # 计算隐含层的输出=每一步的output(组一个list)+一个Final State(单元状态c)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    # 设置 add_output_layer 功能, 添加 output_layer:
    def add_output_layer(self):     # 第三层
        # 从隐含层出来的cell output是一个三维数据，需要转换成二维：shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        # 获取权重矩阵
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        # 获取偏置项
        bs_out = self._bias_variable([self.output_size, ])
        # 计算输出：shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    # 添加 RNN 中剩下的部分:
    def compute_cost(self):     # 计算损失
        # 这里的losses计算的是一个batch里面每一步的loss
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            # 代价=对每一部loss求和，再除以batch_size，得到一个平均值
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            # 对代价cost画像，其实就是每次一个batch训练的平均损失
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    # 定义权重数组，初始为正态分布的随机数
    def _weight_variable(self, shape, name='weights'):      # 定义weight
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)
        # tf.get_variable():获取已存在的变量(要求不仅名字，而且初始化方法等各个参数都一样)，如果不存在，就新建一个。
        # initializer：创建变量的初始化器(默认None)。初始化器也可以是张量，在这种情况下，变量被初始化为该值和形状。

    # 定义偏置项，初始为固定值0.1
    def _bias_variable(self, shape, name='biases'):     # 定义bias
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)




# 训练 LSTMRNN
if __name__ == '__main__':
    # 搭建网络模型
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)

    # 生成会话
    sess = tf.Session()
    # 将所有的图像合并
    merged = tf.summary.merge_all()
    # 保存数据流图
    writer = tf.summary.FileWriter("logs", sess.graph)

    # 对所有变量初始化
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    plt.ion()   # 开启交互模式，保证plt.show()不会阻止主程序的运行
    plt.show()  # 输出图像

    # 训练过程：循环200次，每次去获取一次get_batch()
    for i in range(200):
        # 通过get_batch()获取序列(输入)、结果(预期输出)、xs(整个时间点的data)
        seq, res, xs = get_batch()
        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        # plotting
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')  # 每20个数
        # ndarray.flatten()即返回一个折叠成一维的数组
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)  # 每0.3s刷新一次

        # 打印损失cost
        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)