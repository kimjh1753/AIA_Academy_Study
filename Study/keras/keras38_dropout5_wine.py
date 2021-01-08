# 실습 
# 드랍아웃 적용

import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
print(dataset.DESCR)
print(dataset.feature_names)

# 1. 데이터
x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape)  # (178, 13)
print(y.shape)  # (178,)

# 실습, Dense

# 전처리 알아서 해 / MinMaxScaler, train_test_split
print(np.max(x[0]))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# OneHotEncoding(tensorflow)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x.shape)            # (178, 13)
print(x_train.shape)      # (142, 13)
print(x_test.shape)       # (36, 13) 

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
model = Sequential()
model.add(Dense(13, activation='relu', input_shape=(13,)))  # input = 13
model.add(Dropout(0.2))
model.add(Dense(13, activation='relu'))    
model.add(Dropout(0.2))
model.add(Dense(26, activation='relu'))      
model.add(Dropout(0.2))
model.add(Dense(65, activation='relu'))      
model.add(Dropout(0.2))
model.add(Dense(13, activation='relu'))      
model.add(Dropout(0.2))
model.add(Dense(13, activation='relu'))      
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))   

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=2000, validation_split=0.2, 
          verbose=1, batch_size=13, callbacks=[early_stopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("accruacy : ", accuracy)

# sklearn Dense wine
# loss :  0.0010463970247656107
# accruacy :  1.0

# Dropout 이후
# loss :  0.01913488656282425
# accruacy :  0.9722222089767456