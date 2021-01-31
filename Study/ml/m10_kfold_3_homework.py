# 실습, 과제!!!
# train, test 나눈다음에 train만발리데이션 하지 말고,
# kfold 한 후에 train_test_split 사용

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # Classifier : 분류모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 회귀가 아닌 분류 모델임

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

print(x.shape, y.shape)      # (150, 4) (150, )

# KFold.split : 데이터를 학습 및 테스트 세트로 분할하는 인덱스를 생성
kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델 구성
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

# kfold >> train, test 분리
for train_index, test_index in kfold.split(x) : # 다섯번 반복
 
    # train : test
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # train_test_split >> train, validation 분리
    x_train, x_val, y_train, y_val = \
        train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=47)
    # print(x.shape)          # (150, 4)
    # print(x_train.shape)    # (96, 4)
    # print(x_val.shape)      # (24, 4)
    # print(x_test.shape)     # (30, 4)

    score = cross_val_score(model, x_train, y_train, cv=kfold)
    print('score : ', score)
    
