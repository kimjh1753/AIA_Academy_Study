import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split

#데이터 지정 및 전처리
x = np.load("../study/LPD_COMPETITION/npy/P_project_x4.npy", allow_pickle=True)
y = np.load("../study/LPD_COMPETITION/npy/P_project_y4.npy", allow_pickle=True)
x_pred = np.load('../study/LPD_COMPETITION/npy/pred.npy', allow_pickle=True)

print(x.shape, y.shape, x_pred.shape)   # (48000, 64, 64, 3) (48000, 1000) (72000, 64, 64, 3)

idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),   # 0.1 => acc 하락
    height_shift_range=(-1,1),  # 0.1 => acc 하락
    # rotation_range=40, acc 하락 
    shear_range=0.2)    # 현상유지
    # zoom_range=0.2, acc 하락
    # horizontal_flip=True)

idg2 = ImageDataGenerator()

y = np.argmax(y, axis=1)

x = preprocess_input(x) 
x_pred = preprocess_input(x_pred) 

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,batch_size=128)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = x_pred

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
efficientnetb7 = EfficientNetB7(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])
efficientnetb7.trainable = False
a = efficientnetb7.output
a = Flatten() (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = efficientnetb7.input, outputs = a)

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(patience=10)
reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5)

model.fit_generator(train_generator, epochs=100, validation_data=valid_generator, callbacks=[es, reduce_lr])

model.save('../study/LPD_COMPETITION/h5/challenge3.hdf5')

# predict
result = model.predict(test_generator,verbose=True)
    
print(result.shape)
sub = pd.read_csv('../study/LPD_COMPETITION/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../study/LPD_COMPETITION/answer3.csv',index=False)

