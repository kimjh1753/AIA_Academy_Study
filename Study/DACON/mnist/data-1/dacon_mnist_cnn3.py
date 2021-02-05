import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 늘리는 작업 
from numpy import expand_dims
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam

train = pd.read_csv('../study/DACON/data-1/train.csv')
test = pd.read_csv('../study/DACON/data-1/test.csv')
submission = pd.read_csv('../study/DACON/data-1/submission.csv')

print(train.shape, test.shape) # (2048, 787) (20480, 786) (20480, 2)

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
# tra_di = train['digit'].value_counts()
# print(tra_di.shape) # (10,)

# drop 인덱스
train2 = train.drop(['id', 'digit','letter'],1) # >> x(인덱스 있는 3개 버리기)
test2 = test.drop(['id','letter'],1)  # >> x_pred(인덱스 있는 것 버리기)

train2 = train2.values  # >>> x
test2 = test2.values    # >>> x_pred

train2 = train2.reshape(-1, 28, 28, 1)
print(train2.shape) # (2048, 28, 28, 1)

test2 = test2.reshape(-1, 28, 28, 1)
print(test2.shape) # (20480, 28, 28, 1)

# preprocess
train2 = train2/255.0
test2 = test2/255.0

# ImageDatagenerator & data augmentation >> 데이터 증폭 : 데이터 양을 늘림으로써 오버피팅을 해결할 수 있다.
idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1, 1)) # 이미지 카테고리화(4차원만 가능)
idg2 = ImageDataGenerator() # #ImageDataGenerator 머신러닝
# width_shift_range 좌우로 움직이는 정도:(-1,1) 처음부터 끝까지
# height_shift_range 위아래로 움직이는 정도

'''
sample_data = train2[100].copy()
sample = expand_dims(sample_data,0)
# expand_dims : 차원을 확장시킨다.
sample_datagen = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
sample_generator = sample_datagen.flow(sample, batch_size=1)    #  flow : ImageDataGenerator 디버깅

plt.figure(figsize=(16,10))
for i in range(9) :
    plt.subplot(3, 3, i+1)
    sample_batch = sample_generator.next()
    sample_image = sample_batch[0]
    plt.imshow(sample_image.reshape(28, 28))
plt.show()
'''

# cross validation
skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True) #n_splits 몇 번 반복
# stratified 는 label 의 분포를 유지, 각 fold가 전체 데이터셋을 잘 대표한다.

# 2. Model
reduce_lr = ReduceLROnPlateau(patience=130, factor=0.5, verbose=1)
es = EarlyStopping(patience=150, verbose=1)

val_loss_min = []
val_acc_max = []
result = 0
nth = 0
y = train['digit']

for train_index, test_index in skf.split(train2, y) : # >>> x, y
    path = '../study/dacon/data-1/modelcheckpoint/dacon_mnist_{epoch:02d}-{val_loss:.4f}cp.hdf5'
    mc = ModelCheckpoint(path, save_best_only=True, verbose=1)

    x_train = train2[train_index]
    x_test = train2[test_index]
    y_train = train['digit'][train_index]
    y_test = train['digit'][test_index]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, shuffle=True, random_state=42)

    print(x_train.shape, y_train.shape) # (1796, 28, 28, 1) (1796,)
    print(x_test.shape, y_test.shape)   # (52, 28, 28, 1) (52,)
    print(x_val.shape, y_val.shape)     # (200, 28, 28, 1) (200,)
    
    # 실시간 데이터 증강을 사용해 배치에 대해서 모델을 학습(fit_generator에서 할 것)
    train_generator = idg.flow(x_train, y_train, batch_size=8) #훈련데이터셋을 제공할 제네레이터를 지정
    test_generator = idg2.flow(x_test, y_test, batch_size=8) #테스터데이터셋을 제공할 제네레이터를 지정
    valid_generator = idg2.flow(x_val, y_val) # # validation_data에 넣을 것
    test_generator = idg2.flow(test2, shuffle=False) # # predict(x_test)와 같은 역할

    # 2. Model
    model = Sequential()

    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    # BatchNormalization >> 학습하는 동안 모델이 추정한 입력 데이터 분포의 평균과 분산으로 normalization을 하고자 하는 것
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

    model.add(Conv2D(128,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(10,activation='softmax')) # softmax는 'categorical_crossentropy' 짝꿍

    # 3. Compile, Train    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002, epsilon=None), metrics=['acc']) # y의 acc가 목적
                                                                                   # epsilon : 0으로 나눠지는 것을 피하기 위함
    learning_history = model.fit(train_generator, epochs=1000, validation_data=valid_generator, callbacks=[reduce_lr, es, mc])
    # model.load_weights('../study/dacon/data-1/modelcheckpoint/0203_4_cp.hdf5')

    # 4. Predict
    loss, acc = model.evaluate(test_generator)

    result += model.predict_generator(test_generator,verbose=True)/40 # a += b는 a= a+b, 40은 StratifiedKFold 에서 n_splits=40을 의미
    # predict_generator 예측 결과는 클래스별 확률 벡터로 출력

    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    val_acc_max.append(hist['val_acc'].max())

    nth += 1
    print(nth, '번째 학습을 완료했습니다.') # n_splits 다 돌았는지 확인

    print(val_loss_min, np.mean(val_loss_min))
    print(val_acc_max, np.mean(val_acc_max))
    model.summary()

# submission
submission['digit'] = result.argmax(1)
# submission.to_csv('../study/DACON/data-1/cnn3.csv', index=False)
submission.to_csv('../study/DACON/data-1/cnn3-1.csv', index=False)


# 0.9424999967217446

