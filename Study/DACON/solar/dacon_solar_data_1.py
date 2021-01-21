import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

df = pd.read_csv('../study/DACON/data/train/train.csv', encoding='CP949')
df1 = pd.read_csv('../study/DACON/data/train/merge.csv', encoding='CP949')

print(df)
print(df.info())
print(df.columns) # Index(['Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')

dataset = df.to_numpy()
print(dataset)
print(type(dataset)) # <class 'numpy.ndarray'>
print(dataset.shape) # (52560, 9)

print(df1)
print(df1.info())
print(df1.columns) # Index(['Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')

dataset1 = df1.to_numpy()
print(dataset1)
print(type(dataset1))
print(dataset.shape) # (27216, 9)

x_pred = dataset1
print(x_pred.shape) # (27216, 9)

def split_xy3(dataset, time_steps, y_columns):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_columns

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : ]
        tmp_y = dataset[x_end_number:y_end_number, ]
        x.append(tmp_x)
        y.append(tmp_y) 
    return np.array(x), np.array(y)

time_steps = 336
y_columns = 96
x, y = split_xy3(dataset, time_steps, y_columns)    

print(x)
print(y)
print("x.shape : ", x.shape) # x.shape :  (52129, 336, 9)
print("y.shape : ", y.shape) # y.shape :  (52129, 96, 9)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle=False
)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.8, shuffle=False
)

print(x_train.shape, x_test.shape, x_val.shape) # (33362, 336, 9) (10426, 336, 9) (8341, 336, 9)
print(y_train.shape, y_test.shape, y_val.shape) # (33362, 96, 9) (10426, 96, 9) (8341, 96, 9)

x_pred = x_pred.reshape(81, 336, 9)
print(x_pred)
print(x_pred.shape) # (81, 336, 9)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, Reshape
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, padding='same', input_shape=(336, 9)))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(96*9))
model.add(Reshape([96, 9]))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/dacon_solar_01_{epoch:02d}-{val_loss:.4f}.hdf5'
# 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
hist = model.fit(x_train, y_train, epochs=1, batch_size=1000, 
                 callbacks=[early_stopping], validation_data=(x_val, y_val))

model.save('./DACON/data/train/dacon_Solar0118.h5')

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss : ", result[0])
print("mae : ", result[1])

# y_predict = model.predict(x_test)

# # RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))

# # R2
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

predict = model.predict(x_pred)
print(predict)
print(predict.shape) # (81, 96, 9)

predict = predict.reshape(predict.shape[0]*predict.shape[1], 9)
print(predict)
print(predict.shape) # (7776, 9)

print(type(predict))
dataframe = pd.DataFrame(predict)
print(dataframe)
# dataframe.columns = pd.read_csv("./DACON/data/sample_submission.csv", index_col=0, header=0).columns
# dataframe.index = pd.read_csv("./DACON/data/sample_submission.csv", index_col=0, header=0).index
dataframe.to_csv("./DACON/data/sample_submission.csv", sep=',', index = False, header = False)
'''
# 시각화
plt.rc('font', family='Malgun Gothic')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss ')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss'])
plt.show()
'''

