from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()
x_data = dataset.data
y_data = dataset.target.reshape(-1, 1)
print(x_data.shape, y_data.shape)   # (569, 30) (569, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.8, random_state=66, shuffle=True
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

# [실습] 맹그러!!!
# 최종 skleran의 accuracy_score값으로 결론낼 것!!!

w = tf.Variable(tf.random_normal([30, 1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # sigmoid로 변환

# cost = tf.reduce_mean(tf.square(hypothesis - y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # loss = binary_crossentropy로 변환

train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_train, y:y_train})

        if step % 50 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict ={x:x_test, y:y_test})
    print(" 예측 값 : ", '\n', h, '\n', "원래 값 : ", '\n', c, '\n', "Accuracy : ", a)
    print(" Accuracy : ", accuracy_score(y_test, sess.run(predicted, feed_dict={x:x_test})))

#  Accuracy :  0.95614034
#  Accuracy :  0.956140350877193