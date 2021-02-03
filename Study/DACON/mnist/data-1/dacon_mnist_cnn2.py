import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 늘리는 작업 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

train = pd.read_csv('../study/DACON/data-1/train.csv')
test = pd.read_csv('../study/DACON/data-1/test.csv')
submission = pd.read_csv('../study/DACON/data-1/submission.csv')

print(train.shape, test.shape) # (2048, 787) (20480, 786)

# idx = 318
# img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
# digit = train.loc[idx, 'digit']
# letter = train.loc[idx, 'letter']

# plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
# plt.imshow(img)
# plt.show()

# 1. Data
x = train.drop(['id', 'digit', 'letter'], axis=1).values
x_pred = test.drop(['id', 'letter'], axis=1).values

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

x = x.reshape(-1, 28, 28, 1)
x = x/255
print(x.shape) # (2048, 28, 28, 1)
x_pred = x_pred.reshape(-1, 28, 28, 1)
x_pred = x_pred/255
print(x_pred.shape) # (20480, 28, 28, 1)

y_tmp = train['digit']
y = np.zeros((len(y_tmp), len(y_tmp.unique()))) # np.zeros(shape, dtype, order) >> 0으로 초기화된 넘파이 배열 
for i, digit in enumerate(y_tmp) :
    y[i, digit] = 1
print(y.shape)  # (2048, 10)

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape, y_train.shape) # (1638, 28, 28, 1) (1638, 10)
print(x_val.shape, y_val.shape)   # (410, 28, 28, 1) (410, 10)

# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1, 1)) # 이미지 카테고리화(4차원만 가능)
idg2 = ImageDataGenerator() # #ImageDataGenerator 머신러닝
# width_shift_range 좌우로 움직이는 정도:(-1,1) 처음부터 끝까지
# height_shift_range 위아래로 움직이는 정도

train_generator = idg.flow(x_train,y_train,batch_size=8)
valid_generator = idg2.flow(x_val,y_val)
test_generator = idg2.flow(x_pred,shuffle=False)

# 2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input, BatchNormalization, MaxPooling2D
model = Sequential()
    
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
    
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))
model.add(Dropout(0.3))
    
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))
model.add(Dropout(0.3))
    
model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10,activation='softmax'))

model.summary()

# 3. Compile, Train
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=100, factor=0.5, verbose=1, mode='auto')
es = EarlyStopping(monitor='val_loss', patience=200, verbose=1)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None), metrics=['acc'])
model.fit(train_generator, epochs=10000, validation_data=(valid_generator), callbacks=[reduce_lr, es])

# 4. Evaluate, Predict
# result = 0
# result += model.predict_generator(test_generator,verbose=True)/40
# print("result : ", result)
loss, acc = model.evaluate(x_val, y_val)
print("loss : ", loss)
print("acc : ", acc)

# submission
submission['digit'] = np.argmax(model.predict(test_generator), axis=1)
submission.head()

submission.to_csv('../study/DACON/data-1/cnn2.csv', index=False)

# loss :  0.5172556042671204
# acc :  0.8731707334518433


