import os

import tensorflow as tf



# 初始化变量和模型参数，定义训练闭环中的运算


# 对数几率回归相同的参数和变量初始化

# 数据集中包含4个数据特征和3个可能的输出类（不同类型的花），因此权值矩阵维数是4x3

W = tf.Variable(tf.zeros([4, 3]), name="weights", dtype=tf.float32)
b = tf.Variable(tf.zeros([3], name="bias", dtype=tf.float32))

# 之前的推断现在用于合并
def combine_inputs(X):
    return tf.matmul(X, W) + b

# 使用softmax函数计算输出
def inference(X):
    return tf.nn.softmax(combine_inputs(X))

# 根据训练数据X及其期望输出Y计算损失
# TensorFlow提供了两个版本的sofymax输出计算交叉熵的方法
# 一个版本只应对单个类别专门做了优化，训练数据是一个类别值，就是下面这个函数
# 另一个版本允许用户使用每个样本属于每个类别的概率信息的训练集
def loss(X, Y):
       return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))

# 创建读取文件的基本代码
def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__)+"/"+file_name])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv会将字符串（文本行）转换到具有指定默认值值的由张量列构成的元组中
    # 它还会为每一列设置数据类型
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # 实际上会读取一个文件，并加载一个张量中的batch_size行
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size*50, min_after_dequeue=batch_size)

# 读取或生成训练数据X及其期望输出Y
# 调用时会使用数据集中的默认值，且都是数值型
def inputs():
       sepal_length, sepal_width, petal_length, petal_width, label = \
       read_csv(100, "iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])

       # 将类别名称转换为从0开始计的类别索引
       # 用tf.stack创建一个张量，并利用tf.equal将文件输入与每个可能的值进行比较，
       # 然后利用tf.argmax找到那个张量中值为真的位置，从而有效的将各类别转化为0~2范围的整数
       label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
           tf.equal(label, ["Iris-setosa"]),
           tf.equal(label, ["Iris-versicolor"]),
           tf.equal(label, ["Iris-virginica"])
       ])), 0))

       # 将所关心的所有特征装入单个矩阵中，并对该矩阵转置，使其每行对应一个样本，而每列对应一个特征
       features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))

       return features, label_number
"""
上述代码中，将输入定义为调用read_csv并对所读取的数据进行转换。
为了转换为布尔型，使用tf.equal方法检查属性值是否与某些常量值相等，还利用tf.to_float方法将转换成数值以进行推断。
然后，利用tf.stack方法将所有的布尔值打包进单个张量中。        
"""

# 根据计算的总损失训练或调整模型参数
def train(total_loss):
       learning_rate = 0.01
       return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

# 对训练得到模型进行评估，度量准确率
# 样本输出>0.5de的，则将输出转换为一个正的回答
# 利用tf.equal比较预测结果和实际值是否相等
# 最后利用tf.reduce_mean统计所有正确预测的样本数，除以样本总数，得到正确预测的百分比
def evaluate(sess, X, Y):
       predicted = tf.cast(tf.argmax(inference(X),1))
       print (sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))

# 在一个会话对象中启动数据流图，并运行训练闭环
with tf.Session() as sess:
    # 对所有变量初始化
    tf.global_variables_initializer().run()

    # 获取训练数据X和期望输出Y
    X, Y = inputs()

    # 计算总损失
    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 实际的训练迭代次数
    train_steps = 1000
    for step in range(train_steps):
        sess.run([train_op])
        # 出于调试和学习的目的，查看损失在训练过程中递减的情况
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
