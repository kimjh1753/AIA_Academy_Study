# 61번을 파이프라인으로 구성!!!

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"mal__batch_size" : batches, "mal__optimizer" : optimizers, "mal__drop" : dropout}

hyperparameters = create_hyperparameters()            

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, epochs=1, batch_size=32, verbose=1)

from sklearn.pipeline import Pipeline, make_pipeline
pipe = Pipeline([("scaler", StandardScaler()), ('mal', model2)])

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(pipe, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train)
print(search.best_params_) # {'mal__optimizer': 'rmsprop', 'mal__drop': 0.2, 'mal__batch_size': 40}
print(search.best_estimator_) # Pipeline(steps=[('scaler', StandardScaler()), ('mal',<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000278C3FAB7C0>)])
print(search.best_score_) # 0.949399987856547
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc) # 최종 스코어 :  0.9575999975204468

# keras40_mnist3_dnn
# acc :  0.9724000096321106

# keras61_1_hyperParameter
# 최종 스코어 :  0.9575999975204468