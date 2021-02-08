# 61 카피해서
# model.cv_results를 붙여서 완성

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import dropout

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
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}
hyperparameters = create_hyperparameters()            
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1)
print(search.best_params_) # {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 30}
print(search.best_estimator_) # <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001F4373C20A0>
print(search.best_score_) # 0.9593666791915894
print(search.cv_results_) 
# {'mean_fit_time': array([3.49580288, 2.18963226, 4.14999231, 5.68966993, 2.12418834,
#        3.51691111, 1.87506183, 1.83731103, 1.62180018, 4.07556256]), 'std_fit_time': array([0.34017983, 0.12546115, 0.15482232, 0.03531836, 0.07717849,
#        0.09813157, 0.07288001, 0.05359335, 0.06287254, 0.20233001]), 'mean_score_time': array([1.08281771, 0.6156853 , 1.07380462, 2.02295295, 0.57911825,
#        1.04986819, 0.48838441, 0.58477378, 0.49269295, 1.04358713]), 'std_score_time': array([0.09194561, 0.06353067, 0.03187774, 0.05433139, 0.00577698,
#        0.00768865, 0.00419288, 0.00046719, 0.00422364, 0.00612773]), 'param_optimizer': masked_array(data=['adam', 'rmsprop', 'rmsprop', 'adam', 'rmsprop',
#                    'adadelta', 'rmsprop', 'adam', 'adam', 'rmsprop'],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_drop': masked_array(data=[0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.1, 0.2, 0.3, 0.3],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_batch_size': masked_array(data=[20, 40, 20, 10, 40, 20, 50, 40, 50, 20],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'optimizer': 'adam', 'drop': 0.2, 'batch_size': 20}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 40}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 20}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 10}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 40}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 20}, {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 50}, {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 40}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 50}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 20}], 'split0_test_score': array([0.95389998, 0.94929999, 0.95864999, 0.93405002, 0.95985001,
#        0.2969    , 0.96060002, 0.95744997, 0.95920002, 0.949     ]), 'split1_test_score': array([0.95254999, 0.95615   , 0.95520002, 0.94069999, 0.95835   ,
#        0.3339    , 0.95969999, 0.94870001, 0.95190001, 0.95550001]), 'split2_test_score': array([0.9533    , 0.94945002, 0.95015001, 0.95335001, 0.95655   ,
#        0.28459999, 0.95254999, 0.95910001, 0.95730001, 0.95095003]), 'mean_test_score': array([0.95324999, 0.95163333, 0.95466667, 0.94270001, 0.95825001,
#        0.30513333, 0.95761667, 0.95508333, 0.95613335, 0.95181668]), 'std_test_score': array([0.00055226, 0.00319435, 0.00349054, 0.0080051 , 0.00134908,
#        0.02095175, 0.00360147, 0.00456368, 0.00309229, 0.00272346]), 'rank_test_score': array([ 6,  8,  5,  9,  1, 10,  2,  4,  3,  7])}
acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc) # 최종 스코어 :  0.9639999866485596

# keras40_mnist3_dnn
# acc :  0.9724000096321106

# keras61_1_hyperParameter
# 최종 스코어 :  0.9639999866485596