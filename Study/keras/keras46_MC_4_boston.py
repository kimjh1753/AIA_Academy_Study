import numpy as np

from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)  # (506, 13)
print(y.shape)  # (506,)
print("==========================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 711.0 0.0
print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 전처리(MinMax)
# x = x /711.               
# x = (x - 최소) / (최대 - 최소)
#   = (x - np.mix(x) / (np.max(x) - np.min(x)))
print(np.max(x[0]))

from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)

# print(np.max(x), np.min(x)) # 711.0 0.0 => 1.0 0.0
# print(np.max(x[0]))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)
print(x_train.shape)
print(y_train.shape)

x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, train_size = 0.8, shuffle=True)

print(x_train.shape)
print(x_val.shape)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(1000, activation='relu')(input1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
dense1 = Dense(1000, activation='relu')(dense1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
modelpath = './modelCheckPoint/k46_MC_4_boston_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val),
          verbose=1, callbacks=[es, cp])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss, mae : ", result[0], result[1])

y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6)) # 단위 알아서 찾을 것!

plt.subplot(2, 1, 1)    # 2행 1열중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

# plt.title('Cost loss')    # 한글깨짐 오류 해결할 것 과제1.
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 2행 2열중 두번째
plt.plot(hist.history['mae'], marker='.', c='red', label='mae')
plt.plot(hist.history['val_mae'], marker='.', c='blue', label='val_mae')
plt.grid()

# plt.title('정확도')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# skleran boston Dense
# loss, mae :  15.000182151794434 2.9936561584472656
# RMSE :  3.873007354200969
# mse :  15.00018596569479
# R2 :  0.8205353096631526

# sklearn MC_4_boston
# loss, mae :  7.481131553649902 2.075812339782715
# RMSE :  2.7351656659135797
# mse :  7.481131219992475
# R2 :  0.910494516478944