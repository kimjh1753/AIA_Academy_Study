# 만드러 봐
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)
print(x)
print(y)

# 전처리 알아서 해 / MinMaxScaler, train_test_split
print(np.max(x[0]))

x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size = 0.8, random_state = 66, shuffle = True
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape)
print(y_train.shape)

# 2. 모델 구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
# print(x_test, "의 예측결과", y_pred)

result = model.score(x_test, y_test)
print("model.score : ", result)

acc = accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)

# model = LinearSVC()
# model.score :  0.9736842105263158
# accuracy_score :  0.9736842105263158

# model = SVC()
# model.score :  0.9736842105263158
# accuracy_score :  0.9736842105263158

# model = KNeighborsClassifier()
# model.score :  0.956140350877193
# accuracy_score :  0.956140350877193

# model = LogisticRegression()
# model.score :  0.9649122807017544
# accuracy_score :  0.9649122807017544

# model = DecisionTreeClassifier()
# model.score :  0.9035087719298246
# accuracy_score :  0.9035087719298246

# model = RandomForestClassifier()
# model.score :  0.956140350877193
# accuracy_score :  0.956140350877193

# cancer에서 model들 중 LinearSVC(), SVC()가 성능이 제일 좋다.

# Tensorflow
# acc : 0.9736841917037964