# [실습]
# placeholder 사용

import tensorflow as tf
tf.set_random_seed(66)

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# print(sess.run(W), sess.run(b)) # [0.06524777] [1.4264158]

hypotheses = x_train * W + b

cost = tf.reduce_mean(tf.square(hypotheses - y_train))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for step in range(4361):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())    
    for step in range(2001):
        # sess.run(train, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        if step % 20 == 0:
            # print(step, sess.run(cost, feed_dict={x_train:[1,2,3], y_train:[3,5,7]}), sess.run(W), sess.run(b))
            print(step, cost_val, W_val, b_val)
