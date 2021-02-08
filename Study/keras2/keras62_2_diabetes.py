# 모델 : RandomForestRegressor

from inspect import Parameter
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape, y.shape)      # (442, 10) (442,)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

kfold = KFold(n_splits=3, shuffle=True)

# 2. 모델 구성
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(10,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}
hyperparameters = create_hyperparameters()            
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
model2 = KerasRegressor(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model2, hyperparameters, cv=kfold)
search = RandomizedSearchCV(model2, hyperparameters, cv=kfold)

search.fit(x_train, y_train, verbose=1)
print(search.best_params_) # {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 20}
print(search.best_estimator_) # <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x0000029DAD25DC10>
print(search.best_score_) # -21449.686848958332

result = search.score(x_test, y_test)
print('최종정답률', result) # 최종정답률 -29620.404296875

# xgb
# 최종정답률 0.22732769398337882
# 0.22732769398337882