import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)
print(x)
print(y)


# 전처리 알아서 해 / MinMaxScaler, train_test_split
print(np.max(x[0]))

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

print(x.shape)          # (569, 30)
print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(x_val.shape)      # (114, 30)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, padding='same', input_shape=(30, 1)))
model.add(Flatten())
model.add(Dense(26, activation='relu'))      
model.add(Dense(65, activation='relu'))      
model.add(Dense(13, activation='relu'))      
model.add(Dense(13, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. 컴파일, 훈련
                   # mean_squared_error
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto') 

model.fit(x_train, y_train, epochs=300, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=1, batch_size=13)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss, acc : ", loss, acc)

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])

# sklearn LSTM load_breast_cancer
# loss, acc :  0.07699385285377502 0.9736841917037964
# [[9.9940991e-01]
#  [6.3746900e-14]
#  [9.9935573e-01]
#  [9.9785221e-01]]
# [1 0 1 1]

# conv1d_04_cancer
# loss, acc :  0.8783406019210815 0.9649122953414917
# [[1.0000000e+00]
#  [2.5585153e-10]
#  [1.0000000e+00]
#  [1.0000000e+00]]
# [1 0 1 1]