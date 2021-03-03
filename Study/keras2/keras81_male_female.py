# 나를 찍어서 내가 남자인지 여자인지에 대해
# "남자 acc=0.999" 식으로 출력하는 코드를 만들것
# predict 부분과 결과치는 메일로 보낼 것

import numpy as np
from PIL import Image
from tensorflow.core.protobuf import verifier_config_pb2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, DenseNet121
from tensorflow.keras.models import Model, Sequential, load_model
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

xy_test = train_datagen.flow_from_directory(
    '../data2/image/data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset = "validation"
)

im = Image.open(
    '../data2/image/data/JongHo.jpg'
)

my = np.asarray(im)
my = np.resize(
    my, (128, 128, 3)
)

my = my.reshape(1, 128, 128, 3)
print(my.shape) # (1, 128, 128, 3)

predict = test_datagen.flow(my)

# RGB -> BGR
# from tensorflow.keras.applications.vgg16 import preprocess_input
# my = preprocess_input(my)
# print(my.shape) # (1, 128, 128, 3)

# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
# vgg16.trainable = False

DenseNet121 = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
DenseNet121.trainable = False

# 일반 모델
# model = Sequential()
# model.add(DenseNet121)
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# 함수형 모델
initial_model = DenseNet121
last = initial_model.output
x = Flatten()(last)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs = initial_model.input, outputs = output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es=EarlyStopping(patience=20, verbose=1, monitor='val_loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='val_loss')

model.fit_generator(
    xy_train, 
    steps_per_epoch=10,        
    epochs=100,
    callbacks=[es, rl],
    validation_data=xy_test, 
)

loss, acc = model.evaluate_generator(xy_test, verbose=1)
print('loss : ', loss)
print('acc : ', acc)

pred = np.where(model.predict(predict)>0.5, 1, 0)
print(pred)

pred2 = model.predict(predict)

if pred == 1:
    print('남자일 확률 : ', np.round((pred2*100),2), '% 입니다')
else:
    print('여성일 확률 : ', np.round((pred2*100),2), '% 입니다')    

# keras 67
# JongHo.jpg
# 남성 loss= 0.6932488083839417
# 남성 acc= 0.5406249761581421
# [[1]]
# 남자일 확률 :  [[91.06]] % 입니다

# keras 81
# JongHo.jpg
# loss :  0.5635542869567871
# acc :  0.6858789920806885
# [[1]]
# 남자일 확률 :  [[63.37]] % 입니다

# keras 81
# JongHo.jpg
# loss :  0.5116274952888489
# acc :  0.7233429551124573
# [[1]]
# 남자일 확률 :  [[53.54]] % 입니다
