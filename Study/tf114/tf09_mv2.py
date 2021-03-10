import tensorflow as tf
tf.set_random_seed(66)

x_data = [[73, 51, 65], 
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 70]]
y_data = [[152],
          [185],
          [180],
          [205],
          [142]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight') # x_data의 shape와 w의 shape를 곱한 값이 y_data의 shape와 같아야 한다.
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

# [실습] 앵그러봐
# verbose로 나오는 놈은 cost와 hypotheses

cost = tf.reduce_mean(tf.square(hypothesis - y))

# train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(2001):
#         _, cost_val, hy_val = sess.run([train, cost, hypothesis], feed_dict={x:x_data, y:y_data})
#         if step % 10 == 0:
#             print(step, "cost : ", cost_val, "\n", hy_val)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict = {x:x_data, y:y_data}
    )
    if step % 10 == 0:
        print(step, "cost : ", cost_val, "\n 예측값 : \n", hy_val)        
sess.close()



