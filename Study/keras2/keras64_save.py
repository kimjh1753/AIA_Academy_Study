# 가중치 저장할 것
#1. model.save() 쓸것
#2. pickle 쓸것
# 지금 해결 못함(나중에 해결)

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터 / 전처리
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# 2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='mse')
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

from xgboost import XGBClassifier, XGBRegressor
model2 = XGBClassifier(model2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1, eval_metric=['adam', 'adadelta'], # 회귀 분석 문제는 rmse
          eval_set=[(x_train, y_train), (x_test, y_test)])
print(search.best_params_) # {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 30}
print(search.best_estimator_) # <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001F4373C20A0>
print(search.best_score_) # 0.9593666791915894
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc) # 최종 스코어 :  0.9639999866485596

# keras40_mnist3_dnn
# acc :  0.9724000096321106

# keras61_1_hyperParameter
# 최종 스코어 :  0.9639999866485596

# 저장
import pickle
pickle.dump(search, open('../data/xgb_save/m40.pickle.dat', 'wb'))
print('저장완료')