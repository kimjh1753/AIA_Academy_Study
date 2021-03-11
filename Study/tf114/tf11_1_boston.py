from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

dataset = load_boston()
x_data = dataset.data
y_data = dataset.target.reshape(-1, 1)
print(x_data.shape, y_data.shape) # (506, 13) (506, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])

# [실습] 맹그러!!!
# 최종 skleran의 r2값으로 결론낼 것!!!

w = tf.Variable(tf.random_normal([13, 1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(6001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_train, y:y_train})

        if step % 100 == 0:
            print(step, "\n", "loss : ", cost_val)

    print('R2 : ', r2_score(y_test, sess.run(hypothesis, feed_dict={x:x_test})))

# R2 :  0.6198609466660099   