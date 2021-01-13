import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/삼성전자.csv', encoding = 'CP949', index_col=0, header=0)
print(df)

# 10, 13

df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
df.replace(',','',inplace=True, regex=True)
df = df.astype('float32')

# # 상관 계수 시각화!
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

# plt.show()

df_x = df.iloc[:662, [0,1,2,3,4]]   # (662 ,5)
df_y = df.iloc[:662 ,[3]]   # (662, 1)

# df.replace(',', '', inplace=True)

print(df.shape) # (2400, 14)
print(df_x.shape) # (662, 5)
print(df_y.shape) # (661, 1)

aaa = df_x.to_numpy()
print(aaa)
print(type(aaa)) # <class 'numpy.ndarray'>

bbb = df_y.to_numpy()
print(bbb)
print(type(bbb)) # <class 'numpy.ndarray'>

np.save("../data/npy/삼성전자_x.npy", arr=aaa)
np.save("../data/npy/삼성전자_y.npy", arr=bbb)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        df_x, df_y, train_size = 0.8, random_state=66, shuffle = True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)   
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)        

print(x_train.shape, x_test.shape) # (529, 5, 1) (133, 5, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2, padding='same', input_shape=(5,1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(2000))
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(Dropout(0.2))
model.add(Dense(1))

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelcheckpoint/keras_Samsung_01_{epoch:02d}-{val_loss:.4f}.hdf5'
# 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
hist = model.fit(x_train, y_train, epochs=2000, batch_size=200, callbacks=[early_stopping])

model.save('../data/h5/keras_Samsung_01.h5')

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", result[0])
print("accuracy : ", result[1])

y_predict = model.predict(x_test[-1:])
print(y_predict)

# loss :  28272.9296875
# accuracy :  0.0
# [[49392.93]]
