# 다차원 댄스 모델?
# (n, 32, 32, 3) -> (n, 32, 32, 3)

# 1. 데이터

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])/255.

print(x_train.shape, x_test.shape)   # (50000, 32, 32, 3) (10000, 32, 32, 3)

y_train = x_train
y_test = x_test

print(y_train.shape, y_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

# OnHotEncoding
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(y_train.shape)    # (50000, 10)
# print(y_test.shape)     # (10000, 10)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.layers import Reshape

model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same',
#                  strides=1, input_shape=(32,32,3)))
# model.add(MaxPooling2D(pool_size=2))
model.add(Dense(64, input_shape=(32, 32, 3)))
model.add(Dropout(0.5))
# model.add(Conv2D(1, (2,2), padding='same'))
# model.add(Conv2D(1, (2,2), padding='same'))
model.add(Dense(64))
# model.add(Flatten())
model.add(Dense(3072, activation='relu'))
model.add(Reshape((32, 32, 3)))
model.add(Dense(3, activation='softmax'))

model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
modelpath = '../data/modelcheckpoint/k46_MC_2_cifar10_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, epochs=100, batch_size=2000, validation_split=0.2, 
          verbose=1, callbacks=[es, cp])

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", result[0])
print("accuracy : ", result[1])

# keras cifar10 cnn
# loss :  3.212538480758667
# accuracy :  0.5156999826431274

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

# keras cnn cifar10
# loss :  3.212538480758667
# accuracy :  0.5156999826431274

# keras MC_2_cifar10
# loss :  2.565825939178467
# accuracy :  0.531499981880188
