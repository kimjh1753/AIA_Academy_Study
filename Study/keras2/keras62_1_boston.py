# 모델 : RandomForestRegressor

from inspect import Parameter
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
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

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape, y.shape)      # (506, 13) (506,)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=44)

kfold = KFold(n_splits=3, shuffle=True)

# 2. 모델 구성
def build_model(drop=0.5, optimizer='adam', node1=512, node2=256, node3=128, activation='relu'):
    inputs = Input(shape=(13,), name='input')
    x = Dense(node1, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

def create_hyperparameters():
    batches = [512, 256, 128, 64, 32]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    activation = ['relu', 'linear', 'relu', 'than', 'sigmoid']
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}
hyperparameters = create_hyperparameters()            
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
model2 = KerasRegressor(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(model2, hyperparameters, cv=kfold)
# search = RandomizedSearchCV(model2, hyperparameters, cv=kfold)

search.fit(x_train, y_train, verbose=1)
print(search.best_params_) # {'batch_size': 32, 'drop': 0.1, 'optimizer': 'rmsprop'}
print(search.best_estimator_) # <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000001D805608D60>
print(search.best_score_) # -80.52107493082683

result = search.score(x_test, y_test)
print('최종정답률', result) # 최종정답률 -320.896270751953

# xgb
# 최종정답률 0.7631321786464579
# 0.7631321786464579

