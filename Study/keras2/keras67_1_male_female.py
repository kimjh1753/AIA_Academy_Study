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
    batch_size=32,
    class_mode='binary',
    subset = "training"
)
# Found 1736 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    '../data2/image/data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset = "validation"
)

print(xy_train)
print(xy_test)
# Found 1736 images belonging to 2 classes.

model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es=EarlyStopping(patience=20, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='loss')

history = model.fit_generator(
    xy_train, 
    steps_per_epoch=10,         # steps_per_epoch : 전체 train 수 / batch_size = 160 / 5 = 32
    epochs=100,
    callbacks=[es, rl],
    validation_data=xy_test, 
    validation_steps=4,
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("acc : ", acc[-1])
# acc :  0.5062500238418579
print("val_acc : ", val_acc[:-1])
