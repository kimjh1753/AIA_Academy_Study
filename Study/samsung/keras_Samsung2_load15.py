import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

x_train = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[0]
x_test = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[1]
x_val = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[2]
y_train = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[3]
y_test = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[4]
y_val = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[5]
x_pred = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[6]
print(x_train.shape, x_test.shape, x_val.shape) # (1530, 6, 6) (479, 6, 6) (383, 6, 6)
print(y_train.shape, y_test.shape, y_val.shape) # (1530, 1) (479, 1) (383, 1)
print(x_pred.shape) # (1, 6, 6)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, padding='same', input_shape=(6, 6)))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/keras_Samsung2_01_{epoch:02d}-{val_loss:.4f}.hdf5'
# 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto')
hist = model.fit(x_train, y_train, epochs=10000, batch_size=16, 
                 callbacks=[early_stopping, cp], validation_data=(x_val, y_val))

model.save('./samsung/keras_Samsung0115.h5')

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss : ", result[0])
print("mae : ", result[1])

y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

predict = model.predict(x_pred)
print("1월 15일 삼성주가 예측 : ", predict)

# 시각화
plt.rc('font', family='Malgun Gothic')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss ')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss'])
plt.show()

# 1월 14일(당일날 와서 수정함)
# loss :  1198696.0
# mae :  1064.177734375
# [[89317.336]]

# loss :  867865.5625
# mae :  864.7241821289062
# [[90246.5]]

# loss :  101086.3203125
# mae :  279.9505310058594
# [[88828.9]]

# loss :  83262960.0
# mae :  4924.20556640625
# RMSE :  9124.854
# mse :  83262950.0
# R2 :  0.9997261382149487
# 1월 15일 삼성주가 예측 :  [[90407.69]]