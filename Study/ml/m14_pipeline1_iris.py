from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape, y.shape)      # (150, 4) (150, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=44)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()    
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델
# model = Pipeline([("scaler", MinMaxScaler()), ('malddong', SVC())])
# model = make_pipeline(MinMaxScaler(), SVC())

model = Pipeline([("scaler", StandardScaler()), ('malddong', SVC())])
# model = make_pipeline(StandardScaler(), SVC())

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results)

# MinmaxScaler

# Pipeline
# 0.95

# make_pipeline
# 0.95


# StandardScaler

# Pipeline
# 0.9416666666666667

# make_pipeline
# 0.95
