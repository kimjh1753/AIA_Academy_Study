# 모델 : RandomForestRegressor

from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape, y.shape)      # (506, 13) (506,)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

# parameters = [
#     {'n_estimators' : [100, 200]},
#     {'max_depth' : [6, 8, 10, 12]},
#     {'min_samples_leaf' : [3, 5, 7, 10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}
# ]

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12], 'n_estimators' : [100, 200]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_jobs' : [-1, 2, 4], 'n_estimators' : [100, 200]}
]

# 2. 모델 구성
# model = SVC()
# model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold)
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print("최적의 매개변수 :", model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률', r2_score(y_test, y_pred))

aaa = model.score(x_test, y_test)
print(aaa)

# 최적의 매개변수 : RandomForestRegressor(min_samples_leaf=10)
# 최종정답률 0.29990956161247373
# 0.29990956161247373