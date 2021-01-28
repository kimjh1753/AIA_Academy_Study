# 실습
# 6개의 모델을 완성하라!
# for문 쓸 수 있는 사람은 써봐라!! 후훗

import numpy as np
import warnings
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_wine()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

print(x.shape, y.shape)      # (178, 13) (178,)

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
model = RandomForestClassifier()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('scores : ', scores)

# model = [LinearSVC(), SVC(), KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
# for i in range(6):
#     scores = cross_val_score(model[i], x_train, y_train, cv=kfold)
#     print('scores : ', scores)

'''
# model = LinearSVC()
# scores :  scores :  [0.89655172 0.65517241 0.96428571 0.35714286 0.75      ]

# model = SVC()
# scores :  [0.72413793 0.72413793 0.42857143 0.67857143 0.75      ]

# model = KNeighborsClassifier()
#  [0.68965517 0.55172414 0.53571429 0.71428571 0.64285714]

# model = LogisticRegression()
# scores :   [0.96551724 0.96551724 0.92857143 0.92857143 1.        ]

# model = DecisionTreeClassifier()
# scores :  [0.96551724 0.89655172 0.92857143 0.78571429 0.85714286]

# model = RandomForestClassifier()
# scores : [1.         0.96551724 0.96428571 0.96428571 1.        ]
'''
