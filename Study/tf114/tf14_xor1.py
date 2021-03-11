import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# [7분이면 충분하겠지] 맹그러봐

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2, 1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # sigmoid로 변환

# cost = tf.reduce_mean(tf.square(hypothesis - y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # loss = binary_crossentropy로 변환

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict ={x:x_data, y:y_data})
    print(" 예측 값 : ", '\n', h, '\n', "원래 값 : ", '\n', c, '\n', "Accuracy : ", a)

#  예측 값 :  
#  [[0.5098468 ]
#  [0.50287294]
#  [0.5002181 ]
#  [0.4932434 ]]
#  원래 값 :
#  [[1.]
#  [1.]
#  [1.]
#  [0.]]
#  Accuracy :  0.75