import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

print(x.shape)      # (150, 4)
print(y.shape)      # (150, )
print(x[:5])
print(y)

# 전처리 알아서 해 / MinMaxScaler, train_test_split
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

# 2. 모델 구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

# 3. 컴파일, 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
# print(x_test, "의 예측결과", y_pred)

result = model.score(x_test, y_test)
print("model.score : ", result)

acc = accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)

# model = LinearSVC()
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# model = SVC()
# model.score :  1.0
# accuracy_score :  1.0

# model = KNeighborsClassifier()
# model.score :  1.0
# accuracy_score :  1.0

# model = LogisticRegression()
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# model = DecisionTreeClassifier()
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# model = RandomForestClassifier()
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# iris에서 model들 중 SVC(), KNeighborsClassifier()이 성능이 제일 좋다.   

# Tensorflow
# acc : 1.0