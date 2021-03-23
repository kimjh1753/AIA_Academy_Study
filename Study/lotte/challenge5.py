import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split

# 데이터 지정 및 전처리
x = np.load("../study/LPD_COMPETITION/npy/P_project_x4.npy", allow_pickle=True)
y = np.load("../study/LPD_COMPETITION/npy/P_project_y4.npy", allow_pickle=True)
x_pred = np.load('../study/LPD_COMPETITION/npy/pred.npy', allow_pickle=True)

print(x.shape, y.shape, x_pred.shape)   # (48000, 64, 64, 3) (48000, 1000) (72000, 64, 64, 3)

idg = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1),  
    shear_range=0.2
) 

idg2 = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1),  
    shear_range=0.2
)

# y = np.argmax(y, axis=1)

x = preprocess_input(x) 
x_pred = preprocess_input(x_pred) 

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=49)

train_generator = idg.flow(x_train, y_train, batch_size=64, seed=48)
# seed => random_state
valid_generator = idg2.flow(x_valid, y_valid, batch_size=64, seed=48)
test_generator = idg2.flow(x_pred, batch_size=64, seed=48)

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout, Conv2D
# EfficientNetB5 = EfficientNetB5(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])
# EfficientNetB5.trainable = True
# model = Sequential()
# model.add(EfficientNetB5)
# model.add(Conv2D(1000, 1, padding='same', activation='swish', 
#            kernel_regularizer=regularizers.l2(1e-5),        # 1e-5 
#            activity_regularizer=regularizers.l1(1e-5)))     # 1e-5
# model.add(GlobalAveragePooling2D())
# model.add(GaussianDropout(0.3))                      
# model.add(Flatten())
# model.add(Dense(1000, activation= 'softmax'))

# # MobileNet.summary()
# model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# # 3. 컴파일 훈련
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# # mc1 = tf.train.latest_checkpoint()
# mc = ModelCheckpoint('../study/LPD_COMPETITION/h5/challenge21.hdf5', save_only_true=True, verbose=1)
# es = EarlyStopping(patience=30)
# reduce_lr = ReduceLROnPlateau(patience=15, factor=0.5)

# model.fit_generator(train_generator, epochs=200, validation_data=valid_generator, 
#                     callbacks=[es, reduce_lr, mc], steps_per_epoch= len(x_train) / 64)

# model.load_weights('../study/LPD_COMPETITION/h5/challenge21.hdf5')
model = load_model('../study/LPD_COMPETITION/h5/challenge15.hdf5')

# 수정 예정
# predict

'''
result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - mode')
    pred = model.predict(test_data, steps = len(test_data))
    pred = np.argmax(pred, 1)
    result.append(pred)

    print(f'{tta+1} 번째 제출 파일 저장하는 중')
    temp = np.array(result)
    temp = np.transpose(result)

    temp_mode = stats.mode(temp, axis = 1).mode
    sub.loc[:, 'prediction'] = temp_mode
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)

    temp_count = stats.mode(temp, axis = 1).count
    for i, count in enumerate(temp_count):
        if count < tta/2.:
            print(f'{tta+1} 반복 중 {i} 번째는 횟수가 {count} 로 {(tta+1)/2.} 미만!')
'''
sub = pd.read_csv('../study/LPD_COMPETITION/sample.csv')
save_folder = '../study/LPD_COMPETITION/save_folder'
num = 3

custom = np.zeros([72000, 1000])
result = []
for tta in range(10):
    print(f'{tta+1} 번째 TTA 진행중 - TTA')
    pred = model.predict(x_pred, steps = len(x_pred), verbose = True) # (72000, 1000)
    pred = np.array(pred)
    custom = np.add(custom, pred)
    temp = custom / (tta+1)
    temp_sub = np.argmax(temp, 1)
    temp_percent = np.max(temp, 1)

    count = 0
    i = 0
    for percent in temp_percent:
        if percent < 0.3:
            print(f'{i} 번째 테스트 이미지는 {percent}% 의 정확도를 가짐')
            count += 1
        i += 1
    print(f'TTA {tta+1} : {count} 개가 불확실!')
    result.append(count)
    print(f'기록 : {result}')
    sub.loc[:, 'prediction'] = temp_sub
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(num, (tta+1)), index = False)
