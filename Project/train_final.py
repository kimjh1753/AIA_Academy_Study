import warnings
warnings.filterwarnings(action='ignore')

import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, Activation, LeakyReLU
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.transform import pyramid_expand
from Subpixel import Subpixel

x_train = np.load('../project/celeba-dataset/processed/x_train/x_train.npy')
y_train = np.load('../project/celeba-dataset/processed/y_train/y_train.npy')

x_val = np.load('../project/celeba-dataset/processed/x_val/x_val.npy')
y_val = np.load('../project/celeba-dataset/processed/y_val/y_val.npy')

x_test = np.load('../project/celeba-dataset/processed/x_test/x_test.npy')
y_test = np.load('../project/celeba-dataset/processed/y_test/y_test.npy')

print(x_train.shape, y_train.shape) # (8000, 44, 44, 3) (8000, 176, 176, 3)
print(x_val.shape, y_val.shape)     # (1000, 44, 44, 3) (1000, 176, 176, 3)
print(x_test.shape, y_test.shape)   # (1000, 44, 44, 3) (1000, 176, 176, 3)

x1 = x_train[11] 
x2 = x_val[11]

print(x1.shape, x2.shape) # (44, 44, 3) (44, 44, 3)

# plt.subplot(1, 2, 1)
# plt.imshow(x1)
# plt.subplot(1, 2, 2)
# plt.imshow(x2)
# plt.show()

upscale_factor = 4

inputs = Input(shape=(44, 44, 3))
a = Conv2D(filters=64, 
           kernel_size=5, 
           strides=1, 
           padding='same', 
           activation='tanh')(inputs)
a = Conv2D(filters=64, 
           kernel_size=3, 
           strides=1, 
           padding='same', 
           activation='tanh')(a)
a = Conv2D(filters=32, 
           kernel_size=3, 
           strides=1, 
           padding='same', 
           activation='tanh')(a)
a = Conv2D(filters=16, 
           kernel_size=3, 
           strides=1, 
           padding='same', 
           activation='tanh')(a)
a = Subpixel(filters=3, 
             kernel_size=3, 
             r=upscale_factor,  # r = upscale_factor : (44, 44, 3) -> (176, 176, 3)
             padding='same')(a) 
outputs = LeakyReLU(alpha=0.1)(a)
model = Model(inputs=inputs, outputs=outputs)

model.summary()

cp = ModelCheckpoint('../project/celeba-dataset/models/model-final.h5', monitor='val_loss', verbose=1, save_best_only=True)

model.compile(loss='mse', optimizer='adam', metrics=['mae']) # loss='mse'-> 이미지가 얼마나 같은지, 픽셀 값이 얼마나 같은지 확인하기 위해서 mse사용
history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), verbose=1, callbacks=[cp])

model.save('../project/celeba-dataset/h5/project_final.h5')

result = model.evaluate(x_test, y_test)

print("loss : ", result[0])
print("mae : ", result[1])

# plt.figure(figsize=(10, 6)) # 최초 창의 크기를 가로 10인치 세로 6인치로 설정
# plt.suptitle('epochs = 100, activation = relu') # 제목 설정
# plt.suptitle('epochs = 100, activation = relu, LeakyReLU(alpha=0.1)') # 제목 설정
# plt.suptitle('epochs = 100, activation = linear') # 제목 설정
# plt.suptitle('epochs = 100, activation = linear, LeakyReLU(alpha=0.1)') # 제목 설정
# plt.suptitle('epochs = 100, activation = tanh') # 제목 설정
# plt.suptitle('epochs = 100, activation = tanh, LeakyReLU(alpha=0.1)') # 제목 설정

# plt.subplot(2, 1, 1)    # 2행 1열중 첫번째
# plt.plot(history.history['loss'], marker='.', c='red', label='loss')
# plt.plot(history.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()

# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# plt.subplot(2, 1, 2)    # 2행 1열중 첫번째
# plt.plot(history.history['mae'], marker='.', c='red', label='loss')
# plt.plot(history.history['val_mae'], marker='.', c='blue', label='val_mae')
# plt.grid()

# plt.ylabel('mae')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()


# activation 별로 비교 
# epochs = 100, activation = 'relu'
# loss :  0.0013303515734151006
# mae :  0.022961219772696495

# epochs = 100, activation = 'relu', LeakyReLU(alpha=0.1)
# loss :  0.0012986732181161642
# mae :  0.022124871611595154


# epochs = 100, activation = 'linear'
# loss :  0.0017265279311686754
# mae :  0.025380082428455353

# epochs = 100, activation = 'linear', LeakyReLU(alpha=0.1)
# loss :  0.0017905107233673334
# mae :  0.026167450472712517


# epochs = 100, activation = 'tanh'
# loss :  0.0016838000155985355
# mae :  0.02680797316133976

# epochs = 100, activation = 'than', LeakyReLU(alpha=0.1)
# loss :  0.0016121364897117019
# mae :  0.025504857301712036

# 최종 



