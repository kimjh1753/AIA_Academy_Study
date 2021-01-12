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
from tensorflow.keras.layers import Dense, LSTM, Input
model = Sequential()
model.add(Dense(13, activation='relu', input_shape=(13,)))  # input = 13
model.add(Dense(13, activation='relu'))    
model.add(Dense(26, activation='relu'))      
model.add(Dense(65, activation='relu'))      
model.add(Dense(13, activation='relu'))      
model.add(Dense(13, activation='relu'))      
model.add(Dense(3, activation='softmax'))   

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
modelpath = '../data/modelcheckpoint/k46_MC_8_wine_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor="val_loss", save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, epochs=2000, validation_split=0.2, 
          verbose=1, batch_size=13, callbacks=[early_stopping, cp])

# 4. 평가, 예측
result= model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", result[0])
print("accruacy : ", result[1])

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
plt.plot(hist.history['accuracy'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue', label='val_accuracy')
plt.grid()

# plt.title('정확도')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# sklearn Dense wine
# loss :  0.018188897520303726
# accruacy :  0.9722222089767456

# sklearn MC_7_iris wine 
# loss :  0.01842806115746498
# accruacy :  0.9722222089767456