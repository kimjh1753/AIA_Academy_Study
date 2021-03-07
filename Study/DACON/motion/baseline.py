import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import os
import cv2

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, concatenate, Input, Flatten, Dense
from tensorflow.keras import Model

import warnings
warnings.filterwarnings("ignore")

#폴더 경로를 설정해줍니다.
os.chdir('../study/dacon/data-3/1. open') 

#제공된 데이터들의 리스트를 확인합니다.
print(os.listdir())

train = pd.read_csv('train_df.csv')
submission = pd.read_csv('sample_submission.csv')

print(train.head(2))
# ['baseline.py', 'sample_submission.csv', 'test_imgs', 'test_imgs.zip', 'train_df.csv', 'train_imgs', 'train_imgs.zip']
#                           image       nose_x      nose_y  ...  left_instep_y  right_instep_x  right_instep_y
# 0  001-1-1-01-Z17_A-0000001.jpg  1046.389631  344.757881  ...     826.718013     1063.204067      838.827465
# 1  001-1-1-01-Z17_A-0000003.jpg  1069.850679  340.711494  ...     699.062706     1066.376234      841.499445

print(train.shape) # (4195, 49)

#glob를 활용해 이미지의 경로들을 불러옵니다.
import glob
train_paths = glob.glob('./train_imgs/*.jpg')
test_paths = glob.glob('./test_imgs/*.jpg')
print(len(train_paths), len(test_paths)) # 4195 1600

plt.figure(figsize=(40,20))
count=1

for i in np.random.randint(0,len(train_paths),5):
    
    plt.subplot(5,1, count)
    
    img_sample_path = train_paths[i]
    img = Image.open(img_sample_path)
    img_np = np.array(img)

    keypoint = train.iloc[:,1:49] #위치키포인트 하나씩 확인
    keypoint_sample = keypoint.iloc[i, :]
    
    for j in range(0,len(keypoint.columns),2):
        plt.plot(keypoint_sample[j], keypoint_sample[j+1],'rx')
        plt.imshow(img_np)

    count += 1

train['path'] = train_paths

def trainGenerator():
    for i in range(len(train)):
        img = tf.io.read_file(train['path'][i]) # path(경로)를 통해 이미지 읽기
        img = tf.image.decode_jpeg(img, channels=3) # 경로를 통해 불러온 이미지를 tensor로 변환
        img = tf.image.resize(img, [180,320]) # 이미지 resize 
        target = train.iloc[:,1:49].iloc[i,:] # keypoint 뽑아주기
        
        yield (img, target)

#generator를 활용해 데이터셋 만들기        
train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([180,320,3]),tf.TensorShape([48])))
train_dataset = train_dataset.batch(32).prefetch(1)

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D

#간단한 CNN 모델을 적용합니다.

#간단한 CNN 모델을 적용합니다.

model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(180,320,3)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(48))

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])
model.fit(train_dataset, epochs = 100, verbose=1)

x_test=[]

for test_path in tqdm(test_paths):
    img=tf.io.read_file(test_path)
    img=tf.image.decode_jpeg(img, channels=3)
    img=tf.image.resize(img, [180,320])
    x_test.append(img)

x_test=tf.stack(x_test, axis=0)
print(x_test.shape)

pred=model.predict(x_test)

submission.iloc[:,1:]=pred
print(submission)

submission.to_csv('baseline_submission2.csv', index=False)