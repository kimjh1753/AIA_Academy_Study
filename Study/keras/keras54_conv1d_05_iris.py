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

# ## OneHotEncoding
# from tensorflow.keras.utils import to_categorical
# # from keras.utils.np_utils import to_categorical

# y = to_categorical(y)
# # y_train = to_categorical(y_train)
# # y_test = to_categorical(y_test)
 
print(y)
print(x.shape)  # (150, 4)
print(y.shape)  # (150, 3)

# print(np.max(x[0]))


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# OneHotEncoding(tensorflow)
from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical

y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)    
print(y_train.shape)    

# OneHotEncoding(sklearn)
# from sklearn.preprocessing import OneHotEncoder
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)

# one = OneHotEncoder()
# one.fit(y_train)
# y_train = one.transform(y_train).toarray()
# y_test = one.transform(y_test).toarray()

# print(x.shape, x_train.shape, x_test.shape) # (150, 4) (120, 4) (30, 4) 
# print(x_train.shape)    # (150, 4) -> (120, 4)
# print(y_train.shape)    # (150,) -> (120, 3)       

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape, x_test.shape)
print(y_train.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, padding='same', input_shape=(4, 1)))
model.add(Flatten())
model.add(Dense(26, activation='relu'))      
model.add(Dense(65, activation='relu'))      
model.add(Dense(13, activation='relu'))      
model.add(Dense(13, activation='relu')) 
model.add(Dense(3, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
                   # mean_squared_error
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')              

model.fit(x_train, y_train, validation_split=0.2, 
          epochs=2000, callbacks=[early_stopping], verbose=1, batch_size=13)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss, acc : ", loss, acc)

y_pred = np.array(model.predict(x_train[-5:-1]))
print(y_pred)
print(y_pred.argmax(axis=1))

# sklearn LSTM iris
# loss, acc :  0.08124087750911713 1.0
# [[2.7368754e-10 2.9734897e-03 9.9702650e-01]
#  [6.6601649e-05 9.9899536e-01 9.3812286e-04]
#  [5.4915622e-04 3.3081597e-01 6.6863483e-01]
#  [8.5338586e-05 9.9947685e-01 4.3781847e-04]]
# [2 1 2 1]

# conv1d_05_iris
# loss, acc :  0.09769480675458908 0.9333333373069763
# [[1.4848337e-09 4.8823180e-04 9.9951172e-01]
#  [2.1609017e-05 9.9947983e-01 4.9865205e-04]
#  [4.2622611e-05 5.0528747e-01 4.9466982e-01]
#  [2.7993750e-03 9.9595767e-01 1.2429107e-03]]
# [2 1 1 1]