import warnings
warnings.filterwarnings(action='ignore')

import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, Activation, LeakyReLU
from tensorflow.keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.transform import pyramid_expand
from Subpixel import Subpixel

x_train = np.load('../project/celeba-dataset/processed/x_train/x_train.npy')
y_train = np.load('../project/celeba-dataset/processed/y_train/y_train.npy')

x_val = np.load('../project/celeba-dataset/processed/x_val/x_val.npy')
y_val = np.load('../project/celeba-dataset/processed/y_val/y_val.npy')

print(x_train.shape, y_train.shape) # (18650, 44, 44, 3) (18650, 176, 176, 3)
print(x_val.shape, y_val.shape)     # (5700, 44, 44, 3) (5700, 176, 176, 3)

x1 = x_train[11] 
x2 = x_val[11]

print(x1.shape, x2.shape) # (44, 44, 3) (44, 44, 3)

# plt.subplot(1, 2, 1)
# plt.imshow(x1)
# plt.subplot(1, 2, 2)
# plt.imshow(x2)
# plt.show()

model = load_model('../project/celeba-dataset/h5/project_final.h5', custom_objects={'Subpixel':Subpixel})
                                                                  # custom_objects : 커스텀 마이징 된 모델 클래스를 함께 인식해줌

x_test = np.load('../project/celeba-dataset/processed/x_test/x_test.npy')
y_test = np.load('../project/celeba-dataset/processed/y_test/y_test.npy')

result = model.evaluate(x_test, y_test)
print("loss : ", result[0])
print("mae : ", result[1])

# 최종 결과물 출력

test_idx = 50

# 저해상도 이미지(input)
x1_test = x_test[test_idx]

# 저해상도 이미지 확대시킨 이미지
x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True) # multichannel=True -> 컬러채널 허용

# 모델이 예측한 이미지(output)
y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))

# 정답 이미지
y1_test = y_test[test_idx]

print(x1_test.shape, y1_test.shape) # (44, 44, 3) (176, 176, 3)

# unit8 = 데이터 행렬의 클래스가 uint8 인 이미지를 8 비트 이미지, 이미지 기능은 8 비트 이미지를 배정 밀도로 변환하지 않고 직접 표시 할 수 있습니다.
x1_test = (x1_test * 255).astype(np.uint8) 
x1_test_resized = (x1_test_resized * 255).astype(np.uint8)

y1_test = (y1_test * 255).astype(np.uint8)
y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1) # np.clip => ypred.reshape된 값을 0과 1 사이의 범위로 전환

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
plt.suptitle('Result', fontsize=16)

plt.subplot(1, 4, 1) # 1행 4열중 첫번째
plt.title('input') # 저해상도 이미지
plt.imshow(x1_test)

plt.subplot(1, 4, 2) # 1행 4열중 두번째
plt.title('resized') # input 이미지를 강제로 4배 늘린 이미지
plt.imshow(x1_test_resized)

plt.subplot(1, 4, 3) # 1행 4열중 세번째
plt.title('output') # input 이미지를 Subpixel을 통해 예측한 이미지
plt.imshow(y_pred)

plt.subplot(1, 4, 4) # 1행 4열중 네번째
plt.title('groundtruth') # 원본 이미지
plt.imshow(y1_test)

plt.show()
