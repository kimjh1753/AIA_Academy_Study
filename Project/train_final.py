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
x_val = np.load('../project/celeba-dataset/processed/x_val/x_val.npy')
y_train = np.load('../project/celeba-dataset/processed/y_train/y_train.npy')
y_val = np.load('../project/celeba-dataset/processed/y_val/y_val.npy')

print(x_train.shape, y_train.shape) # (300, 44, 44, 3) (300, 176, 176, 3)
print(x_val.shape, y_val.shape) # (300, 44, 44, 3) (300, 176, 176, 3)

x1 = x_train[0] 
x2 = x_val[0]

print(x1.shape, x2.shape) # (44, 44, 3) (44, 44, 3)

plt.subplot(1, 2, 1)
plt.imshow(x1)
plt.subplot(1, 2, 2)
plt.imshow(x2)
plt.show()

upscale_factor = 4

inputs = Input(shape=(44, 44, 3))
a = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
a = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(a)
a = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(a)
a = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(a)
a = Subpixel(filters=3, kernel_size=3, r=upscale_factor, padding='same')(a) # r = upscale_factor : (44, 44, 3) -> (176, 176, 3)
outputs = LeakyReLU(alpha=0.1)(a)
model = Model(inputs=inputs, outputs=outputs)

model.summary()

cp = ModelCheckpoint('../project/celeba-dataset/models/model-final.h5', monitor='val_loss', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mae']) # loss='mse'-> 이미지가 얼마나 같은지, 픽셀 값이 얼마나 같은지 확인하기 위해서 mse사용
history = model.fit(x_train, y_train, epochs=10, validation_data=[x_val, y_val], verbose=1, callbacks=[cp])

loss = history.history['loss']
mae = history.history['mae']

print("loss : ", loss[-1])  
print("mae : ", mae[-1])    

# x_test_list = sorted(glob.glob(os.path.join(base_path, 'x_test', '*.npy')))
# y_test_list = sorted(glob.glob(os.path.join(base_path, 'y_test', '*.npy')))

# # print(len(x_test_list), len(y_test_list))   # 5649 5649
# # print(x_test_list[0])                       # C:\project\celeba-dataset\processed\x_test\024351.npy

x_test = np.load('../project/celeba-dataset/processed/x_test/x_test.npy')
y_test = np.load('../project/celeba-dataset/processed/y_test/y_test.npy')

print(x_test.shape, y_test.shape) # (300, 44, 44, 3) (300, 176, 176, 3)

test_idx = 10

# 저해상도 이미지(input)
x1_test = x_test[test_idx]

# 저해상도 이미지 확대시킨 이미지
x1_test_resized = pyramid_expand(x1_test, 
                                 4,
                                 multichannel=True) # multichannel=True -> 컬러채널 허용

# 정답 이미지
y1_test = y_test[test_idx]

# 모델이 예측한 이미지(output)
y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))

print(x1_test.shape, y1_test.shape) # (44, 44, 3) (176, 176, 3)

# unit8 = 데이터 행렬의 클래스가 uint8 인 이미지를 8 비트 이미지, 이미지 기능은 8 비트 이미지를 배정 밀도로 변환하지 않고 직접 표시 할 수 있습니다.
x1_test = (x1_test * 255).astype(np.uint8) 
x1_test_resized = (x1_test_resized * 255).astype(np.uint8)

y1_test = (y1_test * 255).astype(np.uint8)
y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1) # ypred.reshape된 값을 0과 1 사이의 범위로 전환

# input 이미지
x1_test = cv2.cvtColor(x1_test, 
                       cv2.COLOR_BGR2RGB) 
                         # COLOR_BGR2RGB -> BGR 사진을 RGB 사진으로 변환, OpenCV에서는 BGR 순서로 저장하고 matplotlib에서는 RGB 순서로 저장이 되어서 변경해주는 역할

# input 이미지를 4배 확대한 이미지
x1_test_resized = cv2.cvtColor(x1_test_resized, 
                               cv2.COLOR_BGR2RGB)
                                 # COLOR_BGR2RGB -> BGR 사진을 RGB 사진으로 변환, OpenCV에서는 BGR 순서로 저장하고 matplotlib에서는 RGB 순서로 저장이 되어서 변경해주는 역할
                                   
# input 이미지로 예측한 이미지(저해상도 -> 고해상도)
y_pred = cv2.cvtColor(y_pred, 
                      cv2.COLOR_BGR2RGB)
                        # COLOR_BGR2RGB -> BGR 사진을 RGB 사진으로 변환, OpenCV에서는 BGR 순서로 저장하고 matplotlib에서는 RGB 순서로 저장이 되어서 변경해주는 역할
                         
# 원본 이미지
y1_test = cv2.cvtColor(y1_test, 
                       cv2.COLOR_BGR2RGB)
                         # COLOR_BGR2RGB -> BGR 사진을 RGB 사진으로 변환, OpenCV에서는 BGR 순서로 저장하고 matplotlib에서는 RGB 순서로 저장이 되어서 변경해주는 역할
                           
plt.figure(figsize=(12, 5))

plt.subplot(1, 4, 1) # 1행 4열중 첫번째
plt.title('input') # 저해상도 이미지
plt.imshow(x1_test, interpolation='bicubic')

plt.subplot(1, 4, 2) # 1행 4열중 첫번째
plt.title('resized') # input 이미지를 강제로 4배 늘린 이미지
plt.imshow(x1_test_resized, interpolation='bicubic')

plt.subplot(1, 4, 3) # 1행 4열중 첫번째
plt.title('output') # input 이미지를 Subpixel을 통해 예측한 이미지
plt.imshow(y_pred, interpolation='bicubic')

plt.subplot(1, 4, 4) # 1행 4열중 첫번째
plt.title('groundtruth') # 원본 이미지
plt.imshow(y1_test, interpolation='bicubic')

plt.show()

# activation 별로 비교
# epochs = 2, activation = 'relu'
# loss :  0.0016763228923082352
# mae :  0.026696009561419487

# epochs = 2, activation = 'relu', LeakyReLU(alpha=0.1)
# loss :  0.0016761336009949446
# mae :  0.026643605902791023


# epochs = 2, activation = 'linear'
# loss :  0.0018620517803356051
# mae :  0.027616102248430252

# epochs = 2, activation = 'linear', LeakyReLU(alpha=0.1)
# loss :  0.006469338666647673
# mae :  0.036886803805828094 


# epochs =2, activation = 'tanh'
# loss :  0.0026562961284071207
# mae :  0.036763615906238556

# epochs = 2, activation = 'than', LeakyReLU(alpha=0.1)
# loss :  0.0027720301877707243
# mae :  0.03674674406647682

# 최종 
# epochs = 30, activation = 'relu', LeakyReLU(alpha=0.1)
# loss :  0.0012984503991901875
# mae :  0.022568069398403168

# epochs = 10, activation = 'relu', LeakyReLU(alpha=0.1)
# loss :  0.001365274772979319
# mae :  0.023278996348381042
