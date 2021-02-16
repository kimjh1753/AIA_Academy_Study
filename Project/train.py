import warnings
warnings.filterwarnings(action='ignore')

import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand
from Subpixel import Subpixel
from DataGenerator import DataGenerator

base_path = r'C:\project\celeba-dataset\processed'

x_train_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
x_val_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

print(len(x_train_list), len(x_val_list)) # 18650 5700
print(x_train_list[0]) # C:\project\celeba-dataset\processed\x_train\000001.npy

x1 = np.load(x_train_list[0])
x2 = np.load(x_val_list[0])

print(x1.shape, x2.shape) # (44, 44, 3) (44, 44, 3)

plt.subplot(1, 2, 1)
plt.imshow(x1)
plt.subplot(1, 2, 2)
plt.imshow(x2)
plt.show()

train_gen = DataGenerator(list_IDs=x_train_list, 
                          labels=None, 
                          batch_size=16, 
                          dim=(44,44), 
                          n_channels=3, 
                          n_classes=None, 
                          shuffle=True)

val_gen = DataGenerator(list_IDs=x_val_list, 
                        labels=None, 
                        batch_size=16, 
                        dim=(44,44), 
                        n_channels=3, 
                        n_classes=None, 
                        shuffle=False)

upscale_factor = 4

inputs = Input(shape=(44, 44, 3))

a = Conv2D(filters=64, 
             kernel_size=5, 
             strides=1, 
             padding='same', 
             activation='relu')(inputs)

a = Conv2D(filters=64, 
             kernel_size=3, 
             strides=1, 
             padding='same', 
             activation='relu')(a)

a = Conv2D(filters=32, 
             kernel_size=3, 
             strides=1, 
             padding='same', 
             activation='relu')(a)

a = Conv2D(filters=upscale_factor**2, 
             kernel_size=3, 
             strides=1, 
             padding='same', 
             activation='relu')(a)

a = Subpixel(filters=3,
               kernel_size=3, 
               r=upscale_factor, 
               padding='same')(a)

outputs = Activation('relu')(a)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

history = model.fit_generator(train_gen, 
                              validation_data=val_gen, 
                              epochs=10, 
                              verbose=1, 
                              callbacks=[ModelCheckpoint(r'C:\PROJECT\celeba-dataset\models\model.h5', 
                                                         monitor='val_loss', 
                                                         verbose=1, 
                                                         save_best_only=True)])

loss = history.history['loss']
mae = history.history['mae']

print("loss : ", loss[-1])  # loss :  0.0021746635902673006
print("mae : ", mae[-1])    # mae :  0.031368955969810486

x_test_list = sorted(glob.glob(os.path.join(base_path, 'x_test', '*.npy')))
y_test_list = sorted(glob.glob(os.path.join(base_path, 'y_test', '*.npy')))

# print(len(x_test_list), len(y_test_list))   # 5649 5649
# print(x_test_list[0])                       # C:\project\celeba-dataset\processed\x_test\024351.npy

test_idx = 21

# 저해상도 이미지(input)
x1_test = np.load(x_test_list[test_idx])

# 저해상도 이미지 확대시킨 이미지
x1_test_resized = pyramid_expand(x1_test, 
                                 4,
                                 multichannel=True) # multichannel=True -> 컬러채널 허용

# 정답 이미지
y1_test = np.load(y_test_list[test_idx])

# 모델이 예측한 이미지(output)
y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))

# print(x1_test.shape, y1_test.shape) # (44, 44, 3) (176, 176, 3)

# unit8 = 데이터 행렬의 클래스가 uint8 인 이미지를 8 비트 이미지, 이미지 기능은 8 비트 이미지를 배정 밀도로 변환하지 않고 직접 표시 할 수 있습니다.
x1_test = (x1_test * 255).astype(np.uint8) 
x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
y1_test = (y1_test * 255).astype(np.uint8)
y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)

# input 이미지
x1_test = cv2.cvtColor(x1_test, 
                       cv2.COLOR_BGR2RGB)

# input 이미지를 4배 확대한 이미지
x1_test_resized = cv2.cvtColor(x1_test_resized, 
                               cv2.COLOR_BGR2RGB)

# 원본 이미지
y1_test = cv2.cvtColor(y1_test, 
                       cv2.COLOR_BGR2RGB)

# input 이미지로 예측한 이미지(저해상도 -> 고해상도)
y_pred = cv2.cvtColor(y_pred, 
                      cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 10))

plt.subplot(1, 4, 1)
plt.title('input')
plt.imshow(x1_test)

plt.subplot(1, 4, 2)
plt.title('resized')
plt.imshow(x1_test_resized)

plt.subplot(1, 4, 3)
plt.title('output')
plt.imshow(y_pred)

plt.subplot(1, 4, 4)
plt.title('groundtruth')
plt.imshow(y1_test)

plt.show()

# epochs = 2
# loss :  0.0021746635902673006
# mae :  0.031368955969810486

# epochs = 10
# loss :  0.001615448622033
# mae :  0.02607521042227745
