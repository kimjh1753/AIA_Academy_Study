import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

dataset = np.loadtxt('../data/csv/data-01-test-score.csv', dtype=float, delimiter=',')    # ???
# [실습] 만들어봐

# 아래값 predict 할 것
# 73, 80, 75, 152
# 93, 88, 93, 185
# 89, 91, 90, 180
# 96, 98, 100, 196
# 73, 66, 70, 142

print(dataset.shape) # (25, 4)

x_data = dataset[:, :-1]
y_data = dataset[:, -1:]

print(x_data.shape, y_data.shape) # (25, 3) (25, 1)

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight') # x_data의 shape와 w의 shape를 곱한 값이 y_data의 shape와 같아야 한다.
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict = {x:x_data, y:y_data}
    )
    if step % 10 == 0:
        print(step, "cost : ", cost_val, "\n 예측값 : \n", hy_val)
# print(sess.run(hypothesis, feed_dict={x:[[73, 80, 75]]})) 
# print(sess.run(hypothesis, feed_dict={x:[[93, 88, 93]]})) 
# print(sess.run(hypothesis, feed_dict={x:[[89, 91, 90]]})) 
# print(sess.run(hypothesis, feed_dict={x:[[89, 91, 90]]})) 
# print(sess.run(hypothesis, feed_dict={x:[[96, 98, 100]]})) 
# print(sess.run(hypothesis, feed_dict={x:[[73, 66, 70]]})) 

print(sess.run(hypothesis, feed_dict={x:[[73, 80, 75],[93, 88, 93],[89, 91, 90],[96, 98, 100],[73, 66, 70]]})) 

# x_pred = [[73, 80, 75],
#           [93, 88, 93],
#           [89, 91, 90],
#           [96, 98, 100],
#           [73, 66, 70]] #(5,3) metrix
# print("===================================================") 
# print(" x_pred : ", "\n", sess.run(hypothesis, feed_dict = {x : x_pred}))


'''
73, 80, 75, 152
93, 88, 93, 185
89, 91, 90, 180
96, 98, 100, 196
73, 66, 70, 142
'''
