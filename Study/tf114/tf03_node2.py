# 뺄셈
# 곱셈
# 나눗셈
# 맹그러라!!!

import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
# node3 = tf.subtract(node1, node2)
# node4 = tf.multiply(node1, node2)
# node5 = tf.divide(node1, node2)
# node6 = tf.mod(node1, node2)

sess = tf.Session()

# print("sess.run(node3) : ", sess.run(node3))
# print("sess.run(node4) : ", sess.run(node4))
# print("sess.run(node5) : ", sess.run(node5))
# print("sess.run(node6) : ", sess.run(node6))

# sess.run(node3) :  -1.0
# sess.run(node4) :  6.0
# sess.run(node5) :  0.6666667
# sess.run(node6) :  2.0

# 1. 덧셈
plus = tf.add(node1, node2)

# 2. 뺄셈
minus = tf.subtract(node1, node2)

# 3. 곱셈
multi = tf.multiply(node1, node2)

# 4. 나눗셈
divi = tf.divide(node1, node2)

# 5. 제곱
pow = tf.pow(node1, node2)

# 6. 나눈 나머지
mod = tf.mod(node1, node2)

# 7. 반대 부호
negative = tf.negative(node1)

# 8. A > B의 True/False
greater = tf.greater(node1, node2)

# 9. A >= B의 True/False
greater_equal = tf.greater_equal(node1, node2)

# 10. A < B의 True/False
less = tf.less(node1, node2)

# 11. A <= B의 True/False
less_equal = tf.less_equal(node1, node2)

# 12. 반대의 참 거짓
logical = tf.logical_not(True)

# 13. 절대값
abs = tf.abs(node1)

print('node1, node2: ', sess.run([node1, node2]))
print('PLUS: ', sess.run(plus))
print('SUBTRACT: ', sess.run(minus))
print('MULTIPLY: ', sess.run(multi))
print('DIVIDE: ', sess.run(divi))
print('POW: ', sess.run(pow))
print('MOD: ', sess.run(mod))
print('NEGATIVE: ', sess.run(negative))
print('GREATER: ', sess.run(greater))
print('GREATER_EQUAL: ', sess.run(greater_equal))
print('LESS: ', sess.run(less))
print('LESS_EQUAL: ', sess.run(less_equal))
print('LOGICAL_NOT: ', sess.run(logical))
print('ABSOLUTE: ', sess.run(abs))

