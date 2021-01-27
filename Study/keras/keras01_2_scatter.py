from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.arange(1, 11)
y = np.array([1,2,4,3,5,5,7,9,8,11])
print("",x,"\n", y)

'''
 [ 1  2  3  4  5  6  7  8  9 10] 
 [ 1  2  3  4  5  5  7  9  8 11]
'''

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))

# 3. 컴파일, 훈련
optimizer = RMSprop(learning_rate=0.01)

model.compile(loss='mse', optimizer=optimizer)
model.fit(x, y, epochs=1000)

y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()