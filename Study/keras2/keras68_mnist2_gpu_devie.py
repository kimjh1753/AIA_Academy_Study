# 인공지능계의 hello world라 불리는 mnist!!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

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
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same',
                 strides=1, input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), kernel_initializer='he_normal')) # relu 계열 -> he_normal  sigmoid, tanh 계열 -> xavier
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (2,2), kernel_regularizer=l1(l1=0.01)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (2,2), strides=2))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

'''
간단한 요약 정리
    
kernel_initializer : He - relu, selu, elu ....
                     Xavier - sigmoid, tahn
(kernel : 가중치(weight))
                     kernel_initializer 를 하게 되면 얼마나 gradient 를 잘 전달 할 수 있느냐와
                     layer 를 얼마나 깊게 쌓을 수 있느냐가 정해짐
                     kernel_initializer 에 존재하는 he 와 xavier 는 각각
                     relu, selu, elu 등과 sigmoid, tahn 등에 사용할 때 적합하다

bias_initializer : bias 는 활성화 함수에 직접적으로 관여하게 되므로 몹시 중요한데,
                   기존에는 0.01 이나 0.1 처럼 매우 작은 양수를 주었으나,
                   학습 방법이 개선 된 지금은 보통 0 으로 초기화를 시킴

kernel_regularizer : 레이어 복잡도에 제한을 두어 가중치가 가장 작은 값을 가지도록 강제함
                     (가중치 값의 분포가 균일해짐)

BatchNormalization : 레이어에 들어가는 batch 값들을 정규화 시킴

Dropout : 훈련 할 때 node 의 갯수를 무작위로 줄임 / 검증할 때엔 dropout 을 하지 않음

Batch, Dropout 과 같이 쓰면 안 좋다고는 하지만 무조건 확정적인 것은 아니며,
실제로도 gan 에서도 함께 쓰이기도 한다                      
'''

# 실습!! 완성하시오!!!
# 지표는 acc   /// 0.985 이상

# 응용
# y_test 10개와 y_pred 10개를 출력하시오

# y_test[:10] = (?,?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?,?)

# 3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
#                      save_best_only=True, mode='auto')
# filepath - 가중치 세이브, 최저점을 찍을 때마다 weight 가 들어간 파일을 만듬
# 세이브 된 최적 가중치를 이용해서 모델 평가, 예측을 좀 더 쉽고 빠르게 할 수 있다
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[es], batch_size=1000)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss : ", result[0]) # 리스트의 첫번째 값은 loss
print("accuracy : ", result[1])  # 리스트이 두번째 값은 acc


# 시각화
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# font_path = 'C:\\Users\\ai\\NanumFontSetup_TTF_GOTHIC\\NanumGothic.ttf'
# fontprop = fm.FontProperties(fname=font_path, size=18)

# plt.figure(figsize=(10, 6)) # (10, 6) 의 면적을 잡음

# plt.subplot(2, 1, 1)    # 2행 1열중 첫번째
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()

# plt.title('손실비용', fontproperties=fontprop)    # 한글깨짐 오류 해결할 것 과제1.
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# plt.subplot(2, 1, 2)    # 2행 2열중 두번째
# plt.plot(hist.history['accuracy'], marker='.', c='red')
# plt.plot(hist.history['val_accuracy'], marker='.', c='blue')
# plt.grid()

# plt.title('정확도', fontproperties=fontprop)
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['accuracy', 'val_accuracy'])

# plt.show()

# keras ModelCheckPoing_mnist
# loss :  0.06158018112182617
# accuracy :  0.9879999756813049

