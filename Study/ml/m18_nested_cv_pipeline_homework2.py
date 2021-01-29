# 모델은 RandomForest 쓰고
# 파이프라인 엮어서 25번 돌리기!!!
# 데이터는 wine

from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
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
dataset = load_wine()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape, y.shape)      # (178, 13) (178,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=44)

parameters = [
    {'mal__n_estimators' : [100, 200], 'mal__max_depth' : [6, 8, 10, 12]},
    {'mal__max_depth' : [6, 8, 10, 12], 'mal__n_estimators' : [100, 200]},
    {'mal__min_samples_leaf' : [3, 5, 7, 10], 'mal__min_samples_split' : [2, 3, 5, 10]},
    {'mal__min_samples_split' : [2, 3, 5, 10], 'mal__min_samples_leaf' : [3, 5, 7, 10]},
    {'mal__n_jobs' : [-1, 2, 4], 'mal__n_estimators' : [100, 200]}
]

# 2. Model
pipe = Pipeline([("scaler", StandardScaler()), ('mal', RandomForestClassifier())])

model = GridSearchCV(pipe, parameters, cv=5)

score = cross_val_score(model, x_train, y_train, cv=5)

print('교차검증점수 : ', score)
# 교차검증점수 :  [1.         0.85714286 1.         1.         1.        ]