import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 늘리는 작업 
from sklearn.model_selection import train_test_split, StratifiedKFold
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
tra_di = train['digit'].value_counts()
print(tra_di.shape) # (10,)

train2 = train.drop(['id', 'digit', 'letter'], 1).values
test2 = test.drop(['id', 'letter'], 1).values

# scaler = MinMaxScaler()
# scaler.fit(train2)
# train2 = scaler.transform(train2)
# test2 = scaler.transform(test2)

train2 = train2.reshape(-1, 28, 28, 1)
train2 = train2/255
print(train2.shape) # (2048, 28, 28, 1)
test2 = test2.reshape(-1, 28, 28, 1)
test2 = test2/255
print(test2.shape) # (20480, 28, 28, 1)

# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1, 1)) # 이미지 카테고리화(4차원만 가능)
idg2 = ImageDataGenerator() # #ImageDataGenerator 머신러닝
# width_shift_range 좌우로 움직이는 정도:(-1,1) 처음부터 끝까지
# height_shift_range 위아래로 움직이는 정도

# 2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input, BatchNormalization, MaxPooling2D
def new_model() :
    model = Sequential()
    model.add(Conv2D(128, kernel_size=3, strides=1, input_shape=(28, 28, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    return model

skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True) #n_splits 몇 번 반복
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=50, factor=0.5, verbose=1, mode='auto')
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

val_loss_min = []
result = 0
nth = 0
y = train['digit']
print(y.shape)

for train_index, valid_index in skf.split(train2, y) :

    x_train = train2[train_index]
    x_val = train2[valid_index]    
    y_train = y[train_index]
    y_val = y[valid_index]

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    
    train_generator = idg.flow(x_train,y_train,batch_size=8)
    valid_generator = idg2.flow(x_val,y_val)
    test_generator = idg2.flow(test2, shuffle=True)

    model = new_model()

    # 3. Compile, Train    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None), metrics=['acc'])
    learning_history = model.fit(train_generator, epochs=2000, validation_data=(valid_generator), callbacks=[reduce_lr, es])

    # 4. Predict
    loss, acc = model.evaluate(test_generator)
    print("loss : ", loss)
    print("acc : ", acc)

    result += model.predict_generator(test_generator,verbose=True)/40
    print("result : ", result)

    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())

    nth += 1
    print(nth, '번째 학습을 완료했습니다.')

    # print(val_loss_min, np.mean(val_loss_min))
    model.summary()

# submission
submission['digit'] = result.argmax(1)
submission.to_csv('../study/DACON/data-1/cnn3.csv', index=False)

# loss :  0.5172556042671204
# acc :  0.8731707334518433

