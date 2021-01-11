# 인공지능계의 hello world라 불리는 mnist!!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

print(x_train[0])
print("y_train[0] : ", y_train[0])
print(x_train[0].shape)                 # (28, 28)

# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255.
# (x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# OnHotEncoding
# 여러분이 하시오!!!!!
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
model.add(Conv2D(10, (2,2), padding='same'))
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
model.add(Dense(10, activation='softmax'))

model.summary()

# 실습!! 완성하시오!!!
# 지표는 acc   /// 0.985 이상

# 응용
# y_test 10개와 y_pred 10개를 출력하시오

# y_test[:10] = (?,?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?,?)

# 3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
modelpath = './modelCheckPoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                     save_best_only=True, mode='auto')
tb = TensorBoard(log_dir='./graph', histogram_freq=0,                               
                 write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es, cp, tb], batch_size=1000)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss : ", result[0]) # 리스트의 첫번째 값은 loss
print("accuracy : ", result[1])  # 리스트이 두번째 값은 acc


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

# keras tensorboard mnist
# loss :  0.06158018112182617
# accuracy :  0.9879999756813049

# 2. TensorBoard log_dir, histogram_freq, write_graph, write_images 알아보기

# log_dir -> 로그 파일을 저장할 디렉토리의 경로이다.

# histogram_freq : 모델의 계층에 대한 활성화 및 가중치 히스토그램을 계산할 빈도이다.(epoch 단위)
# 	         단, 0으로 설정할시 히스토그램 계산 X

# write_graph : TensorBoard에서 그래프를 시각화를 결정함. 
# 	      단, True로 설정할 시 로그 파일 커질 수 있음.

# write_images : TensorBoard에서 이미지로 시각화하기 위해 모델 가중치를 사용할 지 결정하는 것  