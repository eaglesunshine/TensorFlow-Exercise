import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)     #100个随机数列，数据类型为float
# np.random.rand（）：通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。
y_data = x_data*0.1 + 0.3       #目标拟合函数，训练结果weights要接近0.3，bias要接近0.1

###create tensorflow structure start####
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))        #生成随机数列，这里是一维张量（就是一个变量），生成范围-1。0到1.0
bias = tf.Variable(tf.zeros([1]))       #一维零向量，就是初始为0的意思
y = Weights*x_data + bias

loss = tf.reduce_mean(tf.square(y-y_data))
# tf.reduce(input_tensor):计算张量input_tensor的平均值
# tf.square(x):对x里的每一个元素求平方
optimizer = tf.train.GradientDescentOptimizer(0.5)      #创建优化器，参数是学习效率
# tf.train.GradientDescentOptimizer()使用随机梯度下降算法，使参数沿着梯度的反方向，即总损失减小的方向移动，实现更新参数，括号内参数为学习率
train = optimizer.minimize(loss)
# 对loss采用梯度下降法，学习率为0.5

init = tf.global_variables_initializer()        #初始化所有变量
###create tensorflow structure start####

sess = tf.Session()
sess.run(init)      # 激活init方法

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(bias))
