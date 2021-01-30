# 실습 
# RandomSearch, GridSearch와 Pipeline을 엮어라!!
# 모델은 RandomForest

from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape, y.shape)      # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=44)

# parameters = [
#     {'n_estimators' : [100, 200]},
#     {'max_depth' : [6, 8, 10, 12]},
#     {'min_samples_leaf' : [3, 5, 7, 10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}
# ]

parameters = [
    {'mal__n_estimators' : [100, 200], 'mal__max_depth' : [6, 8, 10, 12]},
    {'mal__max_depth' : [6, 8, 10, 12], 'mal__n_estimators' : [100, 200]},
    {'mal__min_samples_leaf' : [3, 5, 7, 10], 'mal__min_samples_split' : [2, 3, 5, 10]},
    {'mal__min_samples_split' : [2, 3, 5, 10], 'mal__min_samples_leaf' : [3, 5, 7, 10]},
    {'mal__n_jobs' : [-1, 2, 4], 'mal__n_estimators' : [100, 200]}
]

# 2. 모델
# pipe = Pipeline([("scaler", MinMaxScaler()), ('mal', RandomForestClassifier())])
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

pipe = Pipeline([("scaler", StandardScaler()), ('mal', RandomForestClassifier())])
# pipe = make_pipeline(StandardScaler(), SVC())

# model = GridSearchCV(pipe, parameters, cv=5)
model = RandomizedSearchCV(pipe, parameters, cv=5)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results)

# GridSearchCV 

# Pipeline MinMaxScaler()
# 0.9385964912280702

# Pipeline StandardScaler()
# 0.956140350877193


# RandomizedSearchCV 

# Pipeline MinMaxScaler()
# 0.9473684210526315

# Pipeline StandardScaler()
# 0.9429824561403509
