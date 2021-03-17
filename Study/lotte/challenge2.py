import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications.resnet50 import preprocess_input

x_train = np.load('../study/LPD_COMPETITION/npy/train_x.npy')
y_train = np.load('../study/LPD_COMPETITION/npy/train_y.npy')
x_val = np.load('../study/LPD_COMPETITION/npy/test_x.npy')
y_val = np.load('../study/LPD_COMPETITION/npy/test_y.npy')
x_pred = np.load('../study/LPD_COMPETITION/npy/pred.npy')

print(x_train.shape, y_train.shape) # (39000, 64, 64, 3) (39000, 1000)
print(x_val.shape, y_val.shape)     # (9000, 64, 64, 3) (9000, 1000)
print(x_pred.shape)                 # (72000, 64, 64, 3)

print(x_train.shape[1:])            # (64, 64, 3)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)
x_pred = preprocess_input(x_pred)

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=2, padding='same', input_shape=(64, 64, 3)))
model.add(Flatten())
model.add(Dense(1000, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[es], batch_size=64)

# predict
result = model.predict(x_pred, verbose=True)
    
print(result.shape)
sub = pd.read_csv('../study/LPD_COMPETITION/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../study/LPD_COMPETITION/answer.csv',index=False)





