import numpy as np

# 1. 데이터
m_x_train = np.load('../data/npy/mnist_x.npy')
m_x_test = np.load('../data/npy/mnist_x.npy')
m_y_train = np.load('../data/npy/mnist_y.npy')
m_y_test = np.load('../data/npy/mnist_y.npy')

x_train = m_x_train.reshape(m_x_train.shape[0], m_x_train.shape[1], m_x_train.shape[2], 1).astype('float32')/255.
x_test = m_x_test.reshape(m_x_test.shape[0], m_x_test.shape[1], m_x_test.shape[2], 1)/255.
# (x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# OnHotEncoding
# 여러분이 하시오!!!!!
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(m_y_train)
y_test = to_categorical(m_y_test)

print(y_train.shape)    # (60000, 10)
print(y_test.shape)     # (10000, 10)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same',
                 strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(9, (2,2), padding='same'))
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
model.add(Dense(10, activation='softmax'))

# 실습!! 완성하시오!!!
# 지표는 acc   /// 0.985 이상

# 응용
# y_test 10개와 y_pred 10개를 출력하시오

# y_test[:10] = (?,?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?,?)

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=300, validation_split=0.2, callbacks=[es], batch_size=2000)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

y_test = np.array(model.predict(x_train[:1]))
print(y_test[:10])
print("============")

y_pred = np.array(model.predict(x_test[:1]))
print(y_pred[:10])

# keras40_mnist2_cnn
# loss :  0.0027267972473055124
# acc :  0.9851999878883362
# [[4.4588484e-18 3.7051479e-22 6.9420164e-21 2.4524782e-14 1.5124747e-20
#   1.0000000e+00 8.7939511e-16 1.3549676e-19 1.0208236e-17 8.6478099e-16]]
# ============
# [[1.2594932e-22 7.7908452e-20 2.1220307e-21 4.4421021e-21 1.4484578e-21
#   6.8512847e-27 6.5198954e-31 1.0000000e+00 5.8166433e-22 6.7756852e-17]]

# load_6_mnist
# loss :  0.0009518300066702068
# acc :  0.994700014591217
# [[6.0841990e-14 4.7501879e-14 5.1697616e-14 4.1312662e-13 5.0693414e-15
#   3.1545032e-14 1.2576443e-15 1.0000000e+00 9.9875254e-14 6.6801626e-13]]
# ============
# [[6.0841990e-14 4.7501879e-14 5.1697616e-14 4.1312662e-13 5.0693414e-15
#   3.1545032e-14 1.2576443e-15 1.0000000e+00 9.9875254e-14 6.6801626e-13]]
