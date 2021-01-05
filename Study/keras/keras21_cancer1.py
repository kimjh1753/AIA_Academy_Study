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

print(x_train.shape)
print(y_train.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(30,))
aaa = Dense(1000, activation='relu')(input1)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
outputs = (Dense(1, activation='sigmoid'))(aaa)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# 3. 컴파일, 훈련
                   # mean_squared_error
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto') 

model.fit(x_train, y_train, epochs=2000, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=1)

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

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])

# loss, acc :  1.0589289665222168 0.9122806787490845
# [[1.       ]
#  [0.0028897]
#  [1.       ]
#  [1.       ]]
# [1 0 1 1]

# y_pred = np.array(model.predict(x_train[-5:-1]))
# print(y_pred)
# print(y_pred.argmax(axis=1))

# loss, acc :  0.5237188339233398 0.9736841917037964
# [[1.0000000e+00]
#  [1.0000000e+00]
#  [1.3445654e-36]
#  [1.0000000e+00]]
# [0 0 0 0]



