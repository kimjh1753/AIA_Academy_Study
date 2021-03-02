import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

print(x.shape, y.shape) # (5,) (5,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# print(model.weights)
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=
array([[ 0.8344002 , -0.9948418 ,  1.0046649 ,  0.46510053]],
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=
array([[ 0.69716525, -0.8087987 ,  0.4159646 ],
       [ 0.12844682,  0.5164604 ,  0.47997332],
       [-0.8359671 ,  0.7755283 ,  0.6449108 ],
       [ 0.14035463, -0.12598628, -0.47676566]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[ 0.99612284,  0.6341238 ],
       [ 0.9598384 , -0.3529107 ],
       [ 0.489828  , -0.5760587 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.7618973],
       [-1.3981314]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

# print(model.trainable_weights)
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=
array([[-1.0269148 , -0.9113284 , -0.04930425, -0.53580886]],
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=
array([[ 0.5478687 ,  0.342924  , -0.84859216],
       [ 0.18884563, -0.49324206,  0.6118903 ],
       [ 0.535553  ,  0.24841988,  0.82618   ],
       [ 0.8066534 , -0.37378436,  0.81171834]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.8831618 , -0.06903315],
       [-0.16472065,  0.269894  ],
       [-0.8576189 ,  1.092778  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-1.23174  ],
       [-1.1513612]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

print(len(model.weights))           # 10 -> layer 하나당 (weights, bias)해서 2개씩 5개의 layer로 구성
print(len(model.trainable_weights)) # 10 -> layer 하나당 (weights, bias)해서 2개씩 5개의 layer로 구성