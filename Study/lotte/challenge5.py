import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications.resnet50 import preprocess_input

x = np.load('../study/LPD_COMPETITION/npy/x_data.npy')
y = np.load('../study/LPD_COMPETITION/npy/y_data.npy')
x_pred = np.load('../study/LPD_COMPETITION/npy/pred.npy')

print(x.shape, y.shape) # (48000, 64, 64, 3) (48000,)
print(x_pred.shape)     # (72000, 64, 64, 3)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
# print(y.shape)  # (48000, 1000)

print(x.shape, y.shape) # (48000, 64, 64, 3) (48000, 1000)
print(x_pred.shape)     # (72000, 64, 64, 3)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True
)

print(x_train.shape[1:])            # (64, 64, 3)

x_pred = preprocess_input(x_pred)

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=2, padding='same', input_shape=(64, 64, 3)))
model.add(Flatten())
model.add(Dense(1000, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[es], batch_size=64)

# predict
result = model.predict(x_pred, verbose=True)
    
print(result.shape)
sub = pd.read_csv('../study/LPD_COMPETITION/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../study/LPD_COMPETITION/answer.csv',index=False)





