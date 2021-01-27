# 만드러 봐
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 1. 데이터
datasets = load_boston()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape)      # (323, 13)
print(y.shape)      # (323,)
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
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
# print(x_test, "의 예측결과", y_pred)

result = model.score(x_test, y_test)
print("model.score : ", result)

acc = r2_score(y_test, y_pred)
print("r2_score : ", acc)

# model = LinearRegression()
# model.score :  0.8005220851783466
# r2_score :  0.8005220851783466

# model = KNeighborsRegressor()
# model.score :  0.7955943632180464
# r2_score :  0.7955943632180464

# model = DecisionTreeRegressor()
# model.score :  0.8776265388624669
# r2_score :  0.8776265388624669

# model = RandomForestRegressor()
# model.score :  0.9276181419671147
# r2_score :  0.9276181419671147

# boston에서 model들 중 RandomForestRegressor()이 성능이 제일 좋다.   

# Tensorflow
# R2 : 0.8810227058319411
