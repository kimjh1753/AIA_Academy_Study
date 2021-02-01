# m31로 만든 0.95 이상의 n_component=? 를 사용하여 
# dnn 모델을 만들것

# mnist dnn보다 성능 좋게 만들어라!!!
# cnn과 비교!!!

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape) # (70000, 28, 28)
print(y.shape) # (70000,)

x = x.reshape(70000, 28*28)
print(x.shape) # (70000, 784)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) # cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
print("cumsum : ", cumsum) 

d = np.argmax(cumsum > 0.95)+1
print("cumsum >= 0.95", cumsum >= 0.95) 
print("d : ", d) # d :  154

pca = PCA(n_components=d)
x = pca.fit_transform(x)
# print(x)
print(x.shape) # (70000, 154)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
print(x_train.shape, x_test.shape) # (56000, 154) (14000, 154)
print(y_train.shape, y_test.shape) # (56000,) (14000,)

# OnHotEncoding
# 여러분이 하시오!!!!!
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)    # (56000, 10)
print(y_test.shape)     # (14000, 10)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(154,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=2000, validation_split=0.2, batch_size=2000, callbacks=[es])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

# 응용
# y_test 10개와 y_pred 10개를 출력하시오

# y_test[:10] = (?,?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?,?)

y_test = np.array(model.predict(x_train[:1]))
print(y_test[:10])
print("============")

y_pred = np.array(model.predict(x_test[:1]))
print(y_pred[:10])

# keras40_mnist2_cnn
# loss :  0.00260396976955235
# acc :  0.9854999780654907
# [[8.6690171e-08 2.8707976e-08 9.1137373e-09 9.6521189e-06 4.6547077e-09
#   9.9998856e-01 7.6187533e-08 5.5741470e-08 1.3864026e-06 2.0224462e-07]]
# ============
# [[7.0327958e-30 2.2413428e-23 6.9391834e-21 9.2217209e-22 5.1841172e-22
#   8.7506048e-26 2.4799229e-27 1.0000000e+00 8.0364114e-26 3.3208760e-17]]

# m32_pca_mnist1_dnn
# loss :  0.005441099405288696
# acc :  0.9704285860061646
# [[6.4362305e-36 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   7.0379887e-34 2.4448430e-32 7.5057366e-36 1.0000000e+00 0.0000000e+00]]
# ============
# [[1.3013622e-32 0.0000000e+00 4.7961759e-35 0.0000000e+00 1.0000000e+00
#   0.0000000e+00 0.0000000e+00 8.4387706e-35 0.0000000e+00 0.0000000e+00]]