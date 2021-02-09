import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,   # 수평방향으로 뒤집기
    vertical_flip=True,     # 수직방향으로 뒤집기
    width_shift_range=0.1,  # 지정된 수평방향 이동 범위내에서 임의로 원본이미지를 이동
    height_shift_range=0.1, # 지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동
    rotation_range=5,       # 지정된 각도 범위내에서 임의로 원본이미지를 회전
    zoom_range=1.2,         # 지정된 확대/축소 범위내에서 임의로 원본이미지를 확대/축소
    shear_range=0.7,        # 밀림 강도 범위내에서 임의로 원본이미지를 변형
    fill_mode='nearest'     # 빈자리를 채워준다
)
test_datagen = ImageDataGenerator(rescale=1./255) # 원래 값 0 ~ 255 -> rescale 후 0 ~ 1
# 테스트는 왜 리스케일만 할까?
# 시험문제는 건들 필요가 없기 때문이다

# flow 또는 flow_from_directory

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train', 
    target_size=(150, 150),
    batch_size=5,                              # 출력되는 y값 개수 설정
    class_mode='binary'
)                                               # (80, 150, 150, 3)
# Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test', 
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)
# Found 120 images belonging to 2 classes.

model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', input_shape=(150, 150, 3)))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15)

history = model.fit_generator(
    xy_train, steps_per_epoch=32, epochs=1000,        # steps_per_epoch : 전체 train 수 / batch_size = 160 / 5 = 32
    validation_data=xy_test, validation_steps=4,
    callbacks=[es, reduce_lr]
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("acc : ", acc[-1])
# acc :  0.5687500238418579
print("val_acc : ", val_acc[:-1])
# val_acc :  [0.44999998807907104, 0.44999998807907104, 0.5, 0.550000011920929, 0.5, 0.550000011920929, 0.44999998807907104, 0.44999998807907104, 0.5, 0.5, 0.5, 0.5, 
#             0.44999998807907104, 0.4000000059604645, 0.5, 0.6000000238418579, 0.699999988079071, 0.3499999940395355, 0.44999998807907104, 0.6000000238418579, 0.30000001192092896, 
#             0.44999998807907104, 0.550000011920929, 0.550000011920929, 0.6000000238418579, 0.699999988079071, 0.5, 0.5, 0.44999998807907104, 0.550000011920929, 0.6000000238418579, 
#             0.44999998807907104, 0.44999998807907104, 0.5, 0.44999998807907104, 0.30000001192092896, 0.699999988079071, 0.6000000238418579, 0.5, 0.6000000238418579, 0.4000000059604645, 
#             0.4000000059604645, 0.550000011920929, 0.5, 0.3499999940395355, 0.6000000238418579, 0.5, 0.5, 0.5, 0.44999998807907104, 0.30000001192092896, 0.5, 0.4000000059604645, 0.6000000238418579, 
#             0.4000000059604645]

# 시각화 할것!!!
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val loss', 'train acc', 'val acc'])
plt.show()



