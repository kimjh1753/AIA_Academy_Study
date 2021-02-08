# lstm 으로 수정할 것
# 파라미터 수정할 것
# 필수 : 노드의 갯수

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, MaxPooling2D, LSTM
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Activation

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)

x_train = x_train.reshape(60000, 28*28, 1)
x_test = x_test.reshape(10000, 28*28, 1)

print(x_train.shape, x_test.shape) # (60000, 28*28, 1) (10000, 28*28, 1)

# 2. 모델
def build_model(drop=0.5, optimizer='adam', node1=256, node2=128, activation='relu'):
    inputs = Input(shape=(28*28, 1), name='input')
    x = LSTM(node1, activation=activation, name='hidden1')(inputs)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(100, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(100, activation=activation, name='hidden4')(x)
    x = Dropout(drop)(x)
    x = Dense(100, activation=activation, name='hidden5')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [256, 128, 64, 32]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    activation = ['relu', 'linear', 'than', 'sigmoid', 'softmax']
    return {"batch_size" : batches, "optimizer" : optimizers, 
            "drop" : dropout, "activation" : activation}
hyperparameters = create_hyperparameters()            
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1)
print(search.best_params_) # {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 32, 'activation': 'sigmoid'}
print(search.best_estimator_) # <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000002837715AC10>
print(search.best_score_) # 0.1620666633049647
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc) # 최종 스코어 :  0.11349999904632568

# keras_40_mnist4_lstm
# acc :  0.11349999904632568
