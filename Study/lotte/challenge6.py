import numpy as np
import pandas as pd
import os
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D, Input, GaussianDropout
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy import stats

#0. 변수
filenum = 5
img_size = 192
batch = 32
seed = 42

#1. 데이터
train_gen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=60,
    zoom_range=0.2,
    fill_mode='nearest',
    validation_split=0.1,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    width_shift_range=0.05,
    height_shift_range=0.05,
    preprocessing_function=preprocess_input

)

# Found 58000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    '../study/LPD_COMPETITION/train_new',
    target_size = (img_size, img_size),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 14000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    '../study/LPD_COMPETITION/train_new',
    target_size = (img_size, img_size),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

#2. 모델
# EfficientNet = EfficientNetB4(include_top=False,weights='imagenet',input_shape=(img_size,img_size,3))
# EfficientNet.trainable = True
# a = EfficientNet.output
# a = Dense(2048, activation= 'swish') (a)
# a = GaussianDropout(0.3) (a)
# a = GlobalAveragePooling2D() (a)
# a = Dense(1000, activation= 'softmax') (a)

# model = Model(inputs = EfficientNet.input, outputs = a)
# model.summary()

# mc = ModelCheckpoint('../study/LPD_COMPETITION/h5/challenge35.hdf5', save_only_true=True, verbose=1)
# es = EarlyStopping(patience=15, verbose=1, mode='auto')
# reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, verbose=1, mode='auto')

# # #3. 컴파일 훈련
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr=0.0005), metrics = ['sparse_categorical_accuracy'])
# model.fit(train_data, steps_per_epoch = (train_data.samples/batch), validation_data= val_data, 
#           validation_steps= (val_data.samples/batch), epochs = 200, callbacks = [es, mc, reduce_lr])

model = load_model('../study/LPD_COMPETITION/h5/challenge35.hdf5')

# Found 72000 images belonging to 1 classes.
test_data = test_datagen.flow_from_directory(
    '../study/LPD_COMPETITION/test_new',
    target_size = (img_size, img_size),
    class_mode = None,
    batch_size = batch,
    shuffle = False
)

#4. 평가 예측

save_folder = '../study/LPD_COMPETITION/save_folder/submit_{0:03}'.format(filenum)
sub = pd.read_csv('../study/LPD_COMPETITION/sample.csv', header = 0)
cumsum = np.zeros([72000, 1000])
count_result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - TTA')
    pred = model.predict(test_data, steps = len(test_data), verbose=True) # (72000, 1000)
    pred = np.array(pred)
    cumsum = np.add(cumsum, pred)
    temp = cumsum / (tta+1)
    temp_sub = np.argmax(temp, 1)
    temp_percent = np.max(temp, 1)
    count = 0
    i = 0
    for percent in temp_percent:
        if percent < 0.3:
            print(f'{i} 번째 테스트 이미지는 {percent}% 의 정확도를 가짐')
            count += 1
        i += 1
    print(f'TTA {tta+1} : {count} 개가 불확실!')
    count_result.append(count)
    print(f'기록 : {count_result}')
    sub.loc[:, 'prediction'] = temp_sub
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)





