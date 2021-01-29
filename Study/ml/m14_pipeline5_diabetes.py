# RandomForest로 구성할것!

from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape, y.shape)      # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=44)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()    
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. Model
# model = Pipeline([("scaler", MinMaxScaler()), ('malddong', RandomForestRegressor())])
# model = make_pipeline(MinMaxScaler(), RandomForestRegressor())

# model = Pipeline([("scaler", StandardScaler()), ('malddong', RandomForestRegressor())])
model = make_pipeline(StandardScaler(), RandomForestRegressor())

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results)

# MinmaxScaler

# Pipeline
# 0.35020917531692186

# make_pipeline
# 0.34805298542613294


# StandardScaler

# Pipeline
# 0.36521218578910564

# make_pipeline
# 0.37127200419501705