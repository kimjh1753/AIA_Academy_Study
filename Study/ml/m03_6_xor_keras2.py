from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델 구성
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, "의 예측결과", y_pred)

result = model.evaluate(x_data, y_data)
print("model.score : ", result[1])

# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 [[0.17603281]
#  [0.536633  ]
#  [0.5700667 ]
#  [0.4919017 ]]
# 1/1 [==============================] - 0s 0s/step - loss: 0.5138 - acc: 1.0000
# model.score :  1.0