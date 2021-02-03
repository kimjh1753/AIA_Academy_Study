import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Data
train = pd.read_csv('../study/DACON/data-1/train.csv')
test = pd.read_csv('../study/DACON/data-1/test.csv')

print(train.shape, test.shape) # (2048, 787) (20480, 786)

# idx = 318
# img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
# digit = train.loc[idx, 'digit']
# letter = train.loc[idx, 'letter']

# plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
# plt.imshow(img)
# plt.show()

x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train/255
print(x_train.shape) # (2048, 28, 28, 1)

y = train['digit']
print(y.shape) # (2048,)
y_train = np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y_train[i, digit] = 1
print(y_train.shape) # (2048, 10)

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
model.add(Conv2D(128, kernel_size=5, strides=1, input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. Compile, Train
# model = create_cnn_model(x_train)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)

x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test/255
print(x_test.shape) # (20480, 28, 28, 1)

# submission
submission = pd.read_csv('../study/DACON/data-1/submission.csv')
submission['digit'] = np.argmax(model.predict(x_test), axis=1)
submission.head()

submission.to_csv('../study/DACON/data-1/baseline.csv', index=False)
