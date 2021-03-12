import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   # False

print(tf.__version__)   # 2.3.1

# 1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, 10)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000, 10)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 2. 모델 구성
# L1.
w1 = tf.compat.v1.get_variable("w1", shape=[3, 3, 1, 32]) # 3, 3, 1, 32 => 3, 3 = 커널 사이즈, 1 = channel(input_dim), 32 = filter
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') # 1,1,1,1 => 1칸씩 띄우기, 1,2,2,1 => 2칸씩 띄우기 (가운데 부분만 바꿈)
print(L1)
# Conv2D(filter, kernel_size, input_shape)      서머리???
# Conv2D(32, (3, 3), input_shape=(28, 28, 1)) 파라미터의 갯수???
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L1)       # shape=(?, 14, 14, 32)

# L2.
w2 = tf.compat.v1.get_variable("w2", shape=[3, 3, 32, 64]) 
L2 = tf.nn.conv2d(L1, w2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.elu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)       # shape=(?, 14, 14, 64)

# L3.
w3 = tf.compat.v1.get_variable("w3", shape=[3, 3, 64, 128]) 
L3 = tf.nn.conv2d(L2, w3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
# L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)       # shape=(?, 14, 14, 128)

# L4.
w4 = tf.compat.v1.get_variable("w4", shape=[3, 3, 128, 64]) 
L4 = tf.nn.conv2d(L3, w4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.selu(L4)
# L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)       # shape=(?, 14, 14, 64)

# Flatten
L_flat = tf.reshape(L4, [-1, 14*14*64])
print("Flatten : ", L_flat)         # shape=(?, 12544)

# L5.
w5 = tf.compat.v1.get_variable("w5", shape=[14*14*64, 64]) #, 
                    #  initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random.normal([64]), name='b5')
L5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5)
# L5 = tf.nn.dropout(L5, keep_prob=0.7)
print(L5)                           # shape=(?, 64)

# L6.
w6 = tf.compat.v1.get_variable("w6", shape=[64, 32]) #, 
                    #  initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random.normal([32]), name='b6')
L6 = tf.nn.selu(tf.matmul(L5, w6) + b6)
# L6 = tf.nn.dropout(L6, keep_prob=0.7)
print(L6)                           # shape=(?, 32)

# L7.
w7 = tf.compat.v1.get_variable("w7", shape=[32, 10]) #, 
                    #  initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random.normal([10]), name='b7')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7)
print("최종출력 : ", hypothesis)    # shape=(?, 10)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))    # categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.000089).minimize(loss)

# 훈련

training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000 / 100 = 600
print(total_batch)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):    # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    
    print("Epoch : ", '%4d' %(epoch + 1),
          'cost = {:.9f}'.format(avg_cost))

print("훈련 끝!!!")          

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1),tf.compat.v1.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print("Acc : ", sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# Epoch :    15 cost = 0.007290391
# 훈련 끝!!!
# Acc :  0.9891

