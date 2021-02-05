import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train = pd.read_csv('../study/DACON/data-1/train.csv')
test = pd.read_csv('../study/DACON/data-1/test.csv')
submission = pd.read_csv('../study/DACON/data-1/submission.csv')

print(train.shape, test.shape) # (2048, 787) (20480, 786)

#distribution of label('digit') 
print(train['digit'].value_counts()) # 각 숫자별 몇개인지
# 2    233
# 5    225
# 6    212
# 4    207
# 3    205
# 1    202
# 9    197
# 7    194
# 0    191
# 8    182
# Name: digit, dtype: int64


# idx = 318
# img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
# digit = train.loc[idx, 'digit']
# letter = train.loc[idx, 'letter']

# plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
# plt.imshow(img)
# plt.show()

# 1. Data
x = train.drop(['id', 'digit', 'letter'], axis=1).values
x = x.reshape(-1, 28, 28, 1)
x = x/255
print(x.shape) # (2048, 28, 28, 1)

y_tmp = train['digit']
y = np.zeros((len(y_tmp), len(y_tmp.unique()))) # np.zeros(shape, dtype, order) >> 0으로 초기화된 넘파이 배열 
for i, digit in enumerate(y_tmp) :
    y[i, digit] = 1
print(y.shape)  # (2048, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape, y_train.shape) # (1638, 28, 28, 1) (1638,)
print(x_test.shape, y_test.shape)   # (410, 28, 28, 1) (410,)

x_pred = test.drop(['id', 'letter'], axis=1).values
x_pred = x_pred.reshape(-1, 28, 28, 1)
x_pred = x_pred/255
print(x_pred.shape) # (20480, 28, 28, 1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input, BatchNormalization, MaxPooling2D

# 2. Model
# def create_cnn_model(x_train):
#     inputs = tf.keras.layers.Input(x_train.shape[1:])

#     bn = tf.keras.layers.BatchNormalization()(inputs)
#     conv = tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu')(bn)
#     bn = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
#     pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

#     bn = tf.keras.layers.BatchNormalization()(pool)
#     conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
#     bn = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
#     pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

#     flatten = tf.keras.layers.Flatten()(pool)

#     bn = tf.keras.layers.BatchNormalization()(flatten)
#     dense = tf.keras.layers.Dense(1000, activation='relu')(bn)

#     bn = tf.keras.layers.BatchNormalization()(dense)
#     outputs = tf.keras.layers.Dense(10, activation='softmax')(bn)

#     model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

#     return model

model = Sequential()
model.add(Conv2D(128, kernel_size=3, strides=1, input_shape=(28, 28, 1), padding='same', activation='relu'))
# model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
# model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
# model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
# model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1000, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1000, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax')) # softmax는 'categorical_crossentropy' 짝꿍

model.summary()
# 3. Compile, Train
# model = create_cnn_model(x_train)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
# es = EarlyStopping(monitor='val_loss', patience=40, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=3000, batch_size=64, validation_split=0.2) #, callbacks=[es])

# 4. Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

# submission
submission['digit'] = np.argmax(model.predict(x_pred), axis=1)
submission.head()

submission.to_csv('../study/DACON/data-1/baseline.csv', index=False)

# loss :  0.9590420126914978
# acc :  0.8146341443061829