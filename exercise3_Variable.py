import tensorflow as tf

state = tf.Variable(0, name='counter')      # 创建变量，名字counter，初始值0
print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)               # 参数相加
update = tf.assign(state, new_value)         # 将new_value值加载到变量state上
# tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number

init = tf.global_variables_initializer()       # must define if difine variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

