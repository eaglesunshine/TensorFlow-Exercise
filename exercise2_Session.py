import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

matrixl1 = tf.constant([[3,3]])     # 常量，只有一行，为[3,3]
matrixl2 = tf.constant([[2],
                        [2]])       # 常量，只有一列，为[2,2]的转置

product = tf.matmul(matrixl1, matrixl2)     #矩阵乘法

# method1
sess = tf.Session()
result = sess.run(product)
print (result)
sess.close()

# method2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)