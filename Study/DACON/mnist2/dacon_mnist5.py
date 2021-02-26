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
#     a = cv2.imread('../study/dacon/data-2/dirty_mnist_2nd/{0:05d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
#     a = np.where((a <= 254) & (a != 0), 0, a)
#     a = cv2.dilate(a, kernel=np.ones((2, 2), np.uint8), iterations=1)
#     image_data = cv2.medianBlur(src=a, ksize= 5)    # #점처럼 놓여있는  noise들을 제거할수있음
#     image_data = cv2.resize(image_data, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
#     image_data = np.array(image_data)
#     image_data = image_data.astype(np.uint8)
#     train_list.append(image_data)

# ===================================================================

# test_list = []

# for i in range(50000, 55000):
#     a = cv2.imread('../study/dacon/data-2/test_dirty_mnist_2nd/{0:05d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
#     a = np.where((a <= 254) & (a != 0), 0, a)
#     a = cv2.dilate(a, kernel=np.ones((2, 2), np.uint8), iterations=1)
#     image_data = cv2.medianBlur(src=a, ksize= 5)    # #점처럼 놓여있는  noise들을 제거할수있음
#     image_data = cv2.resize(image_data, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
#     image_data = np.array(image_data)
#     image_data = image_data.astype(np.uint8)
#     test_list.append(image_data)

# train_img = np.array(train_list)    
# test_img = np.array(test_list)

# np.save('../study/dacon/data-2/train2.npy', train_img)    
# np.save('../study/dacon/data-2/test2.npy', test_img)

x_train = np.load('../study/dacon/data-2/train2.npy')
x_test = np.load('../study/dacon/data-2/test2.npy')
y_train = pd.read_csv('../study/dacon/data-2/dirty_mnist_2nd_answer.csv', index_col=0, header=0)

print(x_train.shape, x_test.shape)  # (50000, 64, 64) (5000, 64, 64)
print(y_train.shape) # (50000, 26)

# Rescale to 0 -> 1 by dividing by max pixel value (255)
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

x_train = x_train.reshape(50000, 64, 64, 1)
x_test = x_test.reshape(5000, 64, 64, 1)

cp = ModelCheckpoint('../study/dacon/data-2/h5/dacon_mnist2.hdf5')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20)

# 2. Model
model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',input_shape=(64, 64 ,1),padding='same'))
model.add(BatchNormalization())
# BatchNormalization >> 학습하는 동안 모델이 추정한 입력 데이터 분포의 평균과 분산으로 normalization을 하고자 하는 것
model.add(Dropout(0.2))
    
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
    
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
    
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(26,activation='softmax')) # softmax는 'categorical_crossentropy' 짝꿍

model.summary()

# 3. Compile, Train    
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc']) # y의 acc가 목적
                                                                      # epsilon : 0으로 나눠지는 것을 피하기 위함
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[cp, rl])

y_pred = model.predict(x_test)
y_pred = np.where(y_pred < 0.5, 0, 1)
print(y_pred)
print(y_pred.shape)

y_submission = pd.read_csv('../study/dacon/data-2/sample_submission.csv')

y_submission.iloc[:, 1:] = y_pred
y_submission.to_csv('../study/dacon/data-2/submission_cnn2.csv', index=False)


