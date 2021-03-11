# [실습]
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, 1)

aaa = OneHotEncoder()
aaa.fit(y_train)
y_train = aaa.transform(y_train).toarray()
y_test = aaa.transform(y_test).toarray()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, 10)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, 10)

x_train = x_train.reshape(60000, 28*28)/255.
x_test = x_test.reshape(10000, 28*28)/255.

print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.random_normal([784, 100], name='weight'))
b1 = tf.Variable(tf.random_normal([100], name='bias'))
layer1 = tf.nn.softmax(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([100, 70], name='weight2'))
b2 = tf.Variable(tf.random_normal([70], name='bias2'))
layer2 = tf.nn.softmax(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([70, 50], name='weight3'))
b3 = tf.Variable(tf.random_normal([50], name='bias3'))
layer3 = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random_normal([50, 10], name='weight4'))
b4 = tf.Variable(tf.random_normal([10], name='bias4'))
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, cost_val, = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict ={x:x_test, y:y_test})
    print(" 예측 값 : ", '\n', h, '\n', "원래 값 : ", '\n', c, '\n', "Accuracy : ", a)
    print(" Accuracy : ", accuracy_score(y_test, sess.run(predicted, feed_dict={x:x_test})))

#  예측 값 :  
#  [[0.01522677 0.10066894 0.07107013 ... 0.07326014 0.15259841 0.15519792]
#  [0.01533827 0.10151103 0.0705472  ... 0.07171634 0.15591489 0.1595735 ]
#  [0.01525766 0.10029317 0.07323711 ... 0.07571833 0.15906616 0.16939291]
#  ...
#  [0.01491411 0.10050035 0.06840883 ... 0.07102605 0.15415153 0.14945643]
#  [0.01443127 0.10303747 0.06671078 ... 0.07610493 0.15875861 0.17050125]
#  [0.01556604 0.10125721 0.07035617 ... 0.07017962 0.1552533  0.15452573]]
#  원래 값 :
#  [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
#  Accuracy :  0.9
#  Accuracy :  0.0