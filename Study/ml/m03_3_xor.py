from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 데이터
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델 구성
model = LinearSVC()   # LinearSVC로는 xor accuracy_score 정확한 값 출력 X, SVC()로 가능함

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, "의 예측결과", y_pred)

result = model.score(x_data, y_data)
print("model.score : ", result)

acc = accuracy_score(y_data, y_pred)
print("accuracy_score : ", acc)

# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 [0 0 0 0]
# model.score :  0.5
# accuracy_score :  0.5

# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 [1 1 1 1]
# model.score :  0.5
# accuracy_score :  0.5

# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 [0 1 1 1]
# model.score :  0.75
# accuracy_score :  0.75