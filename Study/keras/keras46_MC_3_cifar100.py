# 1. 데이터
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])/255.

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', 
                 strides=1, input_shape=(32, 32, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(9, (2,2), padding='valid'))
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
model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
modelpath = '../data/modelcheckpoint/k46_MC_3_cifar100_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, epochs=100, batch_size=2000, validation_split=0.2, 
                 verbose=1, callbacks=[es, cp])

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", result[0])
print("accuracy : ", result[1])

# keras cifar100 cnn
# loss :  5.992544174194336
# accuracy :  0.23280000686645508

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

# keras cifar100 cnn
# loss :  5.992544174194336
# accuracy :  0.23280000686645508

# keras MC_3_cifar100
# loss :  5.701779842376709
# accuracy :  0.2313999980688095
