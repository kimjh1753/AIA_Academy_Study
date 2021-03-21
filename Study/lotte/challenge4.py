import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB7, MobileNet, EfficientNetB4, MobileNetV2, InceptionResNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split

#데이터 지정 및 전처리
x = np.load("../study/LPD_COMPETITION/npy/P_project_x4.npy", allow_pickle=True)
y = np.load("../study/LPD_COMPETITION/npy/P_project_y4.npy", allow_pickle=True)
x_pred = np.load('../study/LPD_COMPETITION/npy/pred.npy', allow_pickle=True)

print(x.shape, y.shape, x_pred.shape)   # (48000, 128, 128, 3) (48000, 1000) (72000, 128, 128, 3)

idg = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1),  
    shear_range=0.2)    # 현상유지

idg2 = ImageDataGenerator()

# y = np.argmax(y, axis=1)

x = preprocess_input(x) 
x_pred = preprocess_input(x_pred) 

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=49)

train_generator = idg.flow(x_train, y_train, batch_size=64, seed=2048)
# seed => random_state
valid_generator = idg2.flow(x_valid, y_valid)
test_generator = x_pred

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout, Conv2D
InceptionResNetV2 = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])
# MobileNet = MobileNet(include_top=True, weights='imagenet')

InceptionResNetV2.trainable = True
a = InceptionResNetV2.output
a = Conv2D(1000, 1, padding='same', activation='swish', 
           kernel_regularizer=regularizers.l2(0.01),       # 1e-5 
           activity_regularizer=regularizers.l1(0.01)) (a) # 1e-5           
a = Flatten() (a)
# a = Activation('swish')(a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = InceptionResNetV2.input, outputs = a)

# MobileNet.summary()
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
mc = ModelCheckpoint('../study/LPD_COMPETITION/h5/challenge8.hdf5', save_only_true=True, verbose=1)
es = EarlyStopping(patience=30)
reduce_lr = ReduceLROnPlateau(patience=15, factor=0.5)

model.fit_generator(train_generator, epochs=200, validation_data=valid_generator, 
                    callbacks=[es, reduce_lr, mc], steps_per_epoch= len(x_train) / 64)

model.save('../study/LPD_COMPETITION/h5/challenge9.hdf5')
# model = load_model('../study/LPD_COMPETITION/h5/challenge9.hdf5')

# predict
result = model.predict(test_generator,verbose=True)
    
print(result.shape)
sub = pd.read_csv('../study/LPD_COMPETITION/sample.csv')
sub['prediction'] = np.argmax(result, axis = 1)
sub.to_csv('../study/LPD_COMPETITION/answer/answer10.csv', index=False)

