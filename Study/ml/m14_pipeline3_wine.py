# RandomForest로 구성할것!

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

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()    
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. Model
# model = Pipeline([("scaler", MinMaxScaler()), ('malddong', RandomForestClassifier())])
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())

# model = Pipeline([("scaler", StandardScaler()), ('malddong', RandomForestClassifier())])
model = make_pipeline(StandardScaler(), RandomForestClassifier())

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results)

# MinmaxScaler

# Pipeline
# 0.8811188811188811

# make_pipeline
# 0.8951048951048951


# StandardScaler

# Pipeline
# 0.8741258741258742

# make_pipeline
# 0.9300699300699301