# 모델 : RandomForestClassfier

from inspect import Parameter
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# 1. 데이터
# x, y = load_iris(return_X_y=True)
# dataset = load_iris()
# x = dataset.data
# y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
# print(x.shape, y.shape)      # (150, 4) (150, )

dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header=0, index_col=0)
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
print(x.shape, y.shape)      # (150, 4) (150, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=44)

# OneHotEncoding(tensorflow)
from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical

y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) # (120, 3)   
print(y_test.shape)  # (30, 3)

kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델 구성
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(4,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}
hyperparameters = create_hyperparameters()            
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model = GridSearchCV(SVC(), parameters, cv=kfold) # 파라미터 100% 가동
search = RandomizedSearchCV(model2, hyperparameters, cv=kfold) # 파라미터 100% 가동

# 3. 훈련
search.fit(x_train, y_train, verbose=1)

# 4. 평가, 예측
search.fit(x_train, y_train, verbose=1)
print(search.best_params_) # {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}
print(search.best_estimator_) # <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E205FD2A90>
print(search.best_score_) # 0.725
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc) # 최종 스코어 : 0.699999988079071

# xgb
# 최종정답률 0.9416666666666667
# 0.9416666666666667
