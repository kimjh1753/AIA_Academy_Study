# 실습
# 남자 여자 구별
# ImageDataGenator을 이용하여 fit_generator 사용해서 완성

import numpy as np
from tensorflow.core.protobuf import verifier_config_pb2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    height_shift_range=0.1,
    width_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../data2/image/data',
    target_size=(128, 128),
    batch_size=2000,
    class_mode='binary',
    subset = "training"
)
# Found 1736 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    '../data2/image/data',
    target_size=(128, 128),
    batch_size=2000,
    class_mode='binary',
    subset = "validation"
)

print(xy_train[0][0].shape)  # (1390, 128, 128, 3)
print(xy_train[0][1].shape)  # (1390,)
print(xy_test[0][0].shape)  # (347, 128, 128, 3)
print(xy_test[0][1].shape)  # (347,)

np.save('../data2/image/npy/keras67_train_x.npy', arr=xy_train[0][0])
np.save('../data2/image/npy/keras67_train_y.npy', arr=xy_train[0][1])
np.save('../data2/image/npy/keras67_test_x.npy', arr=xy_test[0][0])
np.save('../data2/image/npy/keras67_test_y.npy', arr=xy_test[0][1])

x_train = np.load('../data2/image/npy/keras67_train_x.npy')
y_train = np.load('../data2/image/npy/keras67_train_y.npy')
x_test = np.load('../data2/image/npy/keras67_test_x.npy')
y_test = np.load('../data2/image/npy/keras67_test_y.npy')

print(x_train.shape, y_train.shape) # (1390, 128, 128, 3) (1389,)
print(x_test.shape, y_test.shape)   # (347, 128, 128, 3) (347,)