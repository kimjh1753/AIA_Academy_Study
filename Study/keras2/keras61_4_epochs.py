# epochs 100 적용
# validation_split, callback 적용
# early_stopping 5 적용
# Reduce LR 3 적용

# modelcheckpoint 폴더에 hdf5 파일 저장

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import dropout, relu

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# 2. 모델
def build_model(drop=0.5, optimizer='adam', node1=512, node2=256, node3=128, activation='relu'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [512, 256, 128, 64, 32]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    activation = ['relu', 'linear', 'than', 'sigmoid']
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout, "activation" : activation}
hyperparameters = create_hyperparameters()            
model2 = build_model()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
modelpath = '../data/modelcheckpoint/keras61_4_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1, epochs=100, validation_split=0.2, callbacks=[es, cp, reduce_lr])
print(search.best_params_) # {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 30}
print(search.best_estimator_) # <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000207FFEC0220>
print(search.best_score_) # 0.9783166448275248
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc) # 최종 스코어 :  0.9854999780654907

# keras40_mnist3_dnn
# acc :  0.9724000096321106

# keras61_4_epochs
# 최종 스코어 :  0.9854999780654907