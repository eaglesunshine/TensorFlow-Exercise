import tensorflow as tf

# placeholder:占位符——先hold住，以后从外界传入值来代替它
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)
"""
    tf.multiply是点乘，即Returns x * y element-wise,支持broadcasting
    tf.matmul是矩阵乘法，即Multiplies matrix a by matrix b, producing a * b.
"""

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
