# CNN 으로 구성
# 2차원을 4차원으로 늘여서 하시오.

import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (569, 30) (569,)

print(y) # 이진 분류


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True 
)
x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape, x_test.shape, x_val.shape) # (455, 30) (114, 30) (114, 30)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1, 1)

print(x_train.shape, x_test.shape, x_val.shape) # (455, 30, 1, 1) (114, 30, 1, 1) (114, 30, 1, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same',
          strides=1, input_shape=(30, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), padding='same'))
model.add(Flatten())
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
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=32, callbacks=[es], verbose=1, validation_data=(x_val, y_val))

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss, acc : ", loss, acc)

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])

# sklearn Dense load_breast_cancer
# loss, acc :  1.0589289665222168 0.9122806787490845
# [[1.       ]
#  [0.0028897]
#  [1.       ]
#  [1.       ]]
# [1 0 1 1]

# sklearn LSTM load_breast_cancer
# loss, acc :  0.07699385285377502 0.9736841917037964
# [[9.9940991e-01]
#  [6.3746900e-14]
#  [9.9935573e-01]
#  [9.9785221e-01]]
# [1 0 1 1]

# sklearn cnn load_breast_cancer
# loss, acc :  0.4910085201263428 0.9649122953414917
# [[1.0000000e+00]
#  [2.4530623e-06]
#  [1.0000000e+00]
#  [1.0000000e+00]]
# [1 0 1 1]