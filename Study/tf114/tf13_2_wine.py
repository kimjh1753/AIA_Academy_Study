from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

datasets = load_wine()

x_data = datasets.data
y_data = datasets.target.reshape(-1, 1)

aaa = OneHotEncoder()
aaa.fit(y_data)
y_data = aaa.transform(y_data).toarray()

print(x_data.shape, y_data.shape) # (178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.8, random_state=66, shuffle=True
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x = tf.placeholder('float32', [None, 13])
y = tf.placeholder('float32', [None, 3])

w = tf.Variable(tf.random_normal([13, 3]), name='weight' )
b = tf.Variable(tf.random_normal([1, 3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=0.015).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict ={x:x_test, y:y_test})
    print(" 예측 값 : ", '\n', h, '\n', "원래 값 : ", '\n', c, '\n', "Accuracy : ", a)
    print(" Accuracy : ", accuracy_score(y_test, sess.run(predicted, feed_dict={x:x_test})))

#  Accuracy :  0.9907407
#  Accuracy :  0.9722222222222222       


