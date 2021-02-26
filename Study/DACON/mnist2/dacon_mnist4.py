import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 이미지 합치기
# train_list = []

# for i in range(50000):
#     a = cv2.imread('../study/dacon/data-2/dirty_mnist_2nd/{0:05d}.png'.format(i), cv2.IMREAD_COLOR)

#     # 이미지 노이즈 제거
#     a = cv2.fastNlMeansDenoisingColored(a, None, 10, 10, 7, 21)

#     # 이미지 사이즈 줄이기
#     a = cv2.resize(a, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

#     a = a/255

#     train_list.append(a)

# ===================================================================

# test_list = []

# for i in range(50000, 55000):
#     a = cv2.imread('../study/dacon/data-2/test_dirty_mnist_2nd/{0:05d}.png'.format(i), cv2.IMREAD_COLOR)

#     # 이미지 노이즈 제거
#     a = cv2.fastNlMeansDenoisingColored(a, None, 10, 10, 7, 21)

#     # 이미지 사이즈 줄이기
#     a = cv2.resize(a, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

#     a = a/255

#     test_list.append(a)

# train_img = np.array(train_list)    
# test_img = np.array(test_list)

# np.save('../study/dacon/data-2/train.npy', train_img)    
# np.save('../study/dacon/data-2/test.npy', test_img)

x_train = np.load('../study/dacon/data-2/train.npy')
x_test = np.load('../study/dacon/data-2/test.npy')
y_train = pd.read_csv('../study/dacon/data-2/dirty_mnist_2nd_answer.csv', index_col=0, header=0)

print(x_train.shape, x_test.shape)  # (50000, 64, 64, 3) (5000, 64, 64, 3)
print(y_train.shape) # (50000, 26)

from tensorflow.keras.applications import VGG16

model = Sequential()
model.add(VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3)))
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(26,activation='softmax')) # softmax는 'categorical_crossentropy' 짝꿍

cp = ModelCheckpoint('../study/dacon/data-2/h5/dacon_mnist.hdf5')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20)

# 3. Compile, Train    
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002), metrics=['acc']) # y의 acc가 목적
                                                                      # epsilon : 0으로 나눠지는 것을 피하기 위함
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[cp, rl])

y_pred = model.predict(x_test)
y_pred = np.where(y_pred < 0.5, 0, 1)
print(y_pred)
print(y_pred.shape)

y_submission = pd.read_csv('../study/dacon/data-2/sample_submission.csv')

y_submission.iloc[:, 1:] = y_pred
y_submission.to_csv('../study/dacon/data-2/submission_cnn.csv', index=False)


