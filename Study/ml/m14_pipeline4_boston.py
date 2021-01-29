# RandomForest로 구성할것!

from inspect import Parameter
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
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
dataset = load_boston()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape, y.shape)      # (506, 13) (506,)

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
# 0.7434110656608326

# make_pipeline
# 0.754952494914862


# StandardScaler

# Pipeline
# 0.7503114552154868

# make_pipeline
# 0.7401665065872653