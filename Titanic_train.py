import tensorflow as tf

# 初始化变量和模型参数，定义训练闭环中的运算
from envs.tensorflow.Lib import os

# 对数几率回归相同的参数和变量初始化

# 该模型根据乘客的年龄、性别、船票等级来推断他或她是否能幸存下来

W = tf.Variable(tf.zeros([5, 1]), name="weights", dtype=tf.float32)
b = tf.Variable(0, name="bias", dtype=tf.float32)

# 之前的推断现在用于合并
def combine_inputs(X):
    return tf.matmul(X, W) + b

# 新的推断值是将sigmod函数运用到前面的合并值的输出
def inference(X):
    return tf.sigmoid(combine_inputs(X))

# 根据训练数据X及其期望输出Y计算损失
# TensorFlow提供了一个可在单个优化步骤中直接为一个sigmod输出计算交叉熵的方法
def loss(X, Y):
       return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))

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
def inputs():
       passenger_id, survied, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
       read_csv(100, "train.csv",[[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])

       # 转换乘客属性数据(第几等船票)
       is_first_class = tf.to_float(tf.equal(pclass, [1]))
       is_second_class = tf.to_float(tf.equal(pclass, [2]))
       is_third_class = tf.to_float(tf.equal(pclass, [3]))

       gender = tf.to_float(tf.equal(sex, ["female"]))

       #最终将所有特征排列在一个矩阵中，然后对该矩阵转置，使其每行对应一个样本，每列对应一种特征
       features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
       survied = tf.reshape(survied, [100,1])

       return features, survied
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
       predicted = tf.cast(inference(X)>0.5, tf.float32)
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

