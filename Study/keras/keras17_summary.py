import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_dim=1, activation='linear'))
model.add(Dense(7, activation='linear'))
model.add(Dense(6, name='aaa'))
model.add(Dense(1))

model.summary()

# 실습2 + 과제
# ensemble1, 2, 3, 4 에 대해 서머리를 계산하고
# 이해한 것을 과제로 제출할 것
# layer를 만들때 'name' 이란놈에 대해 확인하고 설명할 것
#   얘를 반드시 써야할 때가 있다. 그때를 말하라.


# multi input/output은 fitting할 때, name을 key로 넘겨줘야 함. 
# 따라서 name을 명확하게 명시할 것.