import tensorflow as tf
x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.3], tf.float32)

hypothesis = W * x + b
# ...
# print('hypothesis : ', ????)

# [실습]
# 1. sess.run()
# 2. InteractiveSession()
# 3. .eval(session=sess)
# hypothesis를 출력한느 코드를 만드시오!

# sess = tf.Session()
sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
sess.run(tf.compat.v1.global_variables_initializer())
aaa =  sess.run(hypothesis)
print("aaa : ", aaa) # [2.2086694]
sess.close()

# sess = tf.InteractiveSession()
sess = tf.compat.v1.InteractiveSession()
# sess.run(tf.global_variables_initializer())
sess.run(tf.compat.v1.global_variables_initializer())
bbb = hypothesis.eval()  # 변수쩜 이발
print("bbb : ", bbb) # [2.2086694]
sess.close()

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("ccc : ", ccc) # [2.2086694]
sess.close()
