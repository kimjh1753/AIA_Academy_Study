import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

x = np.load('../study/samsung/삼성전자_x.npy')
y = np.load('../study/samsung/삼성전자_y.npy')

print(x.shape) # (662, 11)
print(y.shape) # (662, 1)

size = 10

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])  # [subset]과의 차이는?
    print(type(aaa))
    return np.array(aaa)
x_data = split_x(x, size)
print(x_data.shape) # (657, 7, 11)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state=311, shuffle = True
)
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)   
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
       
print(x_train.shape, x_test.shape, x_val.shape)  # (423, 11, 1) (133, 11, 1) (106, 11, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2, padding='same', input_shape=(11, 1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/keras_Samsung_01_{epoch:02d}-{val_loss:.4f}.hdf5'
# 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
hist = model.fit(x_train, y_train, epochs=2000, batch_size=100, callbacks=[early_stopping, cp], validation_data=(x_val, y_val))

model.save('../study/samsung/keras_Samsung.h5')

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", result[0])
print("mae : ", result[1])

x_predict = np.array([[89800,91200,89100,89700,-0.99, 34161101, 
                       4557102, -1781416, -2125136, -579352, -2190214]])
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(1,11,1)
y_predict = model.predict(x_predict)
print(y_predict)

# 첫번째
# loss :  376809.1875
# mae :  548.606201171875
# [[89878.22]]

# 두번째
# loss : 267404.28125
# mae : 387.7078857421875
# [[90241.65]]

# 1일마다
# loss :  212548.390625
# mae :  356.6076354980469
# [[88792.44]]

# 7일마다
# loss :  320810.8125
# mae :  488.83782958984375
# [[89817.484]]

# 30일마다
# loss :  340002.90625
# mae :  475.162109375
# [[87813.086]]
