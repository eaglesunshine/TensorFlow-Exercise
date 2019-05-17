import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 生成随机数
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 偏置值推荐不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + bias  # 预测值, inputs是300x1的矩阵， weights是1xout的矩阵, bias是1xout的矩阵
    if activation_function is None:  # 直接输出
        outputs = Wx_plus_b
    else:  # 使用f（x）计算
        outputs = activation_function(Wx_plus_b)
    return outputs



"""
1.tf.random_normal
tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
random_normal: 正态分布随机数，均值mean,标准差stddev

2.tf.truncated_normal
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
truncated_normal:截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2stddev,mean+2stddev]范围内的随机数

3.tf.random_uniform
tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)
random_uniform:均匀分布随机数，范围为[minval,maxval]
"""