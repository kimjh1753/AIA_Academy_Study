# 실습
# 6개의 모델을 완성하라!
# for문 쓸 수 있는 사람은 써봐라!! 후훗

import numpy as np
import warnings
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

print(x.shape, y.shape)      # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델 구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

# scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print('scores : ', scores)

'''
# model = LinearSVC()
# scores :  [0.96703297 0.87912088 0.92307692 0.69230769 0.95604396]

# model = SVC()
# scores :  [0.9010989  0.93406593 0.86813187 0.95604396 0.9010989 ]

# model = KNeighborsClassifier()
# [0.91208791 0.95604396 0.93406593 0.89010989 0.96703297]

# model = LogisticRegression()
# scores :  [0.95604396 0.95604396 0.95604396 0.91208791 0.92307692]

# model = DecisionTreeClassifier()
# scores :  [0.91208791 0.95604396 0.93406593 0.95604396 0.95604396]

# model = RandomForestClassifier()
# scores :  [0.95604396 0.96703297 0.95604396 0.96703297 0.95604396]
'''

model = [LinearSVC(), SVC(), KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
for i in range(6):
    scores = cross_val_score(model[i], x_train, y_train, cv=kfold)
    print('scores : ', scores)

# scores :  [0.8021978  0.93406593 0.94505495 0.91208791 0.91208791]
# scores :  [0.92307692 0.91208791 0.92307692 0.85714286 0.92307692]
# scores :  [0.92307692 0.95604396 0.92307692 0.92307692 0.92307692]
# scores :  [0.92307692 0.94505495 0.93406593 0.96703297 0.93406593]
# scores :  [0.94505495 0.93406593 0.9010989  0.93406593 0.94505495]
# scores :  [0.97802198 0.96703297 0.95604396 0.93406593 0.95604396]
