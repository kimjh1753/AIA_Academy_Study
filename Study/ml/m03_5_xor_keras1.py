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
model.add(Dense(1, input_dim=2, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, "의 예측결과", y_pred)

result = model.evaluate(x_data, y_data)
print("model.score : ", result[1])

# acc = accuracy_score(y_data, y_pred)
# print("accuracy_score : ", acc)

# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 [[0.48342043]
#  [0.67017096]
#  [0.46726945]
#  [0.65570027]]
# 1/1 [==============================] - 0s 0s/step - loss: 0.7220 - acc: 0.7500
# model.score :  0.75 -> 히든 레이어(activation = 'relu' 포함)가 없어서 xor score 출력 값 안나옴