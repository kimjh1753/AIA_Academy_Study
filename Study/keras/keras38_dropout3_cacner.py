# 실습 
# 드랍아웃 적용

# keras21_cancer1.py를 다중분류로 코딩하시오.

import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)
print(x)
print(y)


# 전처리 알아서 해 / MinMaxScaler, train_test_split
print(np.max(x[0]))

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)
x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

## OneHotEncoding(tensorflow)
from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical

# y = to_categorical(y)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)

# print(x_train.shape)    # (569, 30) -> (455, 30)    
# print(y_train.shape)    # (569,) -> (455, 2)

# OneHotEncoding(sklearn)
from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
y_val = one.transform(y_val).toarray()

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(30,)))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
                   # mean_squared_error
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, epochs=2000, validation_data=(x_val, y_val), 
          callbacks=[early_stopping], verbose=1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss, acc : ", loss, acc)

# 실습1. acc 0.985 이상 올릴 것
# 실습2. predict 출력해볼것.

'''
y[-5:-1] = ? 0 아니면 1
y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])
'''
# y_pred = model.predict(x_test[-5:-1])
# print(y_pred)
# print(y_test[-5:-1])

# loss, acc :  1.0386602878570557 0.9736841917037964
# [[1.1606958e-33 1.0000000e+00]
#  [1.0000000e+00 3.1406053e-10]
#  [2.1355669e-22 1.0000000e+00]
#  [5.2167696e-15 1.0000000e+00]]
# [[0. 1.]
#  [1. 0.]
#  [0. 1.]
#  [0. 1.]]

y_pred = np.array(model.predict(x_train[-5:-1]))
print(y_pred)
print(y_pred.argmax(axis=1))

# loss, acc :  1.61712646484375 0.9736841917037964
# [[0.0000000e+00 1.0000000e+00]
#  [1.5326388e-29 1.0000000e+00]
#  [1.0000000e+00 0.0000000e+00]
#  [6.9364208e-24 1.0000000e+00]]
# [1 1 0 1]

# Dropout 이후
# loss, acc :  0.3748251795768738 0.9736841917037964
# [[4.5158355e-32 1.0000000e+00]
#  [2.2585601e-08 1.0000000e+00]
#  [1.0000000e+00 9.0088050e-18]
#  [1.5365019e-07 9.9999988e-01]]
# [1 1 0 1]