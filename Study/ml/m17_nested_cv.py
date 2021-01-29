from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# 1. 데이터
# x, y = load_iris(return_X_y=True)

# dataset = load_iris()
# x = dataset.data
# y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
# print(x.shape, y.shape)      # (150, 4) (150, )

datasets = pd.read_csv('../data/csv/iris_sklearn.csv', header=0, index_col=0)
x = datasets.iloc[:, :-1]
y = datasets.iloc[:, -1]
print(x.shape, y.shape)      # (150, 4) (150, )

# ccc = datasets.to_numpy()
# print(ccc)
# print(type(ccc)) # <class 'numpy.ndarray'>

# ddd = datasets.values
# print(ddd)
# print(type(ddd)) # <class 'numpy.ndarray'>

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]

# 2. 모델 구성
# model = SVC()
model = GridSearchCV(SVC(), parameters, cv=kfold) # 파라미터 100% 가동

score = cross_val_score(model, x_train, y_train, cv=kfold)

print('교차검증점수 : ', score)
# 교차검증점수 :  [1.         1.         1.         0.83333333 0.66666667]

'''
# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print("최적의 매개변수 :", model.best_estimator_) 

y_pred = model.predict(x_test)
print('최종정답률', accuracy_score(y_test, y_pred))

aaa = model.score(x_test, y_test)
print(aaa)

datasets = datasets.to_csv('../data/csv/iris_sklearn(gridSearch1).csv', sep=',')

# 최적의 매개변수 : SVC(C=1, kernel='linear')
# 최종정답률 0.9583333333333334
# 0.9583333333333334
'''