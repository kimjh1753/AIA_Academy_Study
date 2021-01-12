import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

print(x.shape)      # (150, 4)
print(y.shape)      # (150, )
print(x[:5])
print(y)

# 전처리 알아서 해 / MinMaxScaler, train_test_split

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

# # OneHotEncoding(tensorflow)
# from tensorflow.keras.utils import to_categorical
# # from keras.utils.np_utils import to_categorical

# y = to_categorical(y)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)

# print(x_train.shape)    
# print(y_train.shape)    

# OneHotEncoding(sklearn)
from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
y_val = one.transform(y_val).toarray()

print(x_train.shape)    # (150, 4) -> (120, 4)
print(y_train.shape)    # (150,) -> (120, 3)       

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
aaa = Dense(1000, activation='relu')(input1)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
outputs = (Dense(3, activation='softmax'))(aaa)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# 3. 컴파일, 훈련
                   # mean_squared_error
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')              
modelpath = '../data/modelcheckpoint/k46_MC_7_iris_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2000, 
                 callbacks=[early_stopping, cp], verbose=1)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss, accuracy : ", result[0], result[1])

y_pred = np.array(model.predict(x_train[-5:-1]))
print(y_pred)
print(y_pred.argmax(axis=1))

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

# keras Dense iris
# loss, acc :  0.041334398090839386 1.0
# [[8.5435504e-14 3.4071996e-05 9.9996591e-01]
#  [2.0802061e-31 9.9999607e-01 3.9425763e-06]
#  [1.5944058e-04 1.3975589e-01 8.6008465e-01]
#  [1.9400661e-26 9.9997354e-01 2.6421489e-05]]
# [2 1 2 1]

# keras MC_7_iris
# loss, accuracy :  0.05247119441628456 1.0
# [[1.2714019e-13 7.1310060e-05 9.9992871e-01]
#  [1.2116942e-24 9.9999988e-01 6.8928479e-08]
#  [1.4658807e-03 1.6673796e-01 8.3179617e-01]
#  [2.5097811e-24 9.9999988e-01 8.7393254e-08]]
# [2 1 2 1]