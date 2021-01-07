# hist를 이용하여 그래프를 그리시오.
# loss, val_loss

# 1. 데이터
import numpy as np
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (442, 10) (442,)
print(x[:5])
print(y[:10])
 
print(np.max(x), np.min(x)) # 0.198787989657293 -0.137767225690012
print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)
x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(10,))
dense1 = Dense(200, activation='relu')(input1)
dense1 = Dense(200, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dense(500, activation='relu')(input1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
hist = model.fit(x_train, y_train, batch_size=32, epochs=2000, validation_data=(x_val, y_val), 
                 verbose=1, callbacks=[early_stopping])
print(hist)
print(hist.history.keys())

print(hist.history['loss'])

# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val loss'])
plt.show()

