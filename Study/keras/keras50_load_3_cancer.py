import numpy as np

# 1. 데이터
x_data = np.load('../data/npy/cancer_x.npy')
y_data = np.load('../data/npy/cancer_y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=0.8, random_state=66, shuffle=True 
)
x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, train_size=0.8, random_state=66, shuffle=True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape, x_test.shape, x_val.shape) # (455, 30) (114, 30) (114, 30)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1, 1)

print(x_train.shape, x_test.shape, x_val.shape) # (455, 30, 1, 1) (114, 30, 1, 1) (114, 30, 1, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same',
          strides=1, input_shape=(30, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), padding='same'))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=64, callbacks=[es], verbose=1, validation_data=(x_val, y_val))

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss, acc : ", loss, acc)

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])

# sklearn cnn load_breast_cancer
# loss, acc :  0.4910085201263428 0.9649122953414917
# [[1.0000000e+00]
#  [2.4530623e-06]
#  [1.0000000e+00]
#  [1.0000000e+00]]
# [1 0 1 1]

# load_3_cancer
# loss, acc :  0.2692721486091614 0.9649122953414917
# [[1.0000000e+00]
#  [1.1206896e-13]
#  [1.0000000e+00]
#  [9.9999857e-01]]
# [1 0 1 1]