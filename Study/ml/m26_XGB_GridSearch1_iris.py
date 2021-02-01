# 데이터별로 5개 만든다.

from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape, y.shape)      # (150, 4) (150, )
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01],
     "max_depth":[4,5,6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01]},
    {"max_depth": [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate" : [0.1, 0.001, 0.5],
     "max_depth": [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1],
     "colsample_bylevel" : [0.6, 0.7, 0.9]}
]
n_jobs = -1

# 2. 모델 구성
# model = SVC()
model = GridSearchCV(XGBClassifier(n_jobs=-1, eval_metric='mlogloss'), parameters, cv=kfold) 

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print("최적의 매개변수 :", model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률', accuracy_score(y_test, y_pred))

aaa = model.score(x_test, y_test)
print(aaa)

# 최적의 매개변수 : RandomForestClassifier(min_samples_leaf=3, min_samples_split=5)
# 최종정답률 0.9583333333333334
# 0.9583333333333334

# xgb
# 최종정답률 0.9333333333333333
# 0.9333333333333333