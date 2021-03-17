import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    rescale=1./255,
    horizontal_flip=True,   # 수평방향으로 뒤집기
    vertical_flip=True,     # 수직방향으로 뒤집기
    width_shift_range=0.1,  # 지정된 수평방향 이동 범위내에서 임의로 원본이미지를 이동
    height_shift_range=0.1, # 지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동
    rotation_range=5,       # 지정된 각도 범위내에서 임의로 원본이미지를 회전
    zoom_range=1.2,         # 지정된 확대/축소 범위내에서 임의로 원본이미지를 확대/축소
    shear_range=0.7,        # 밀림 강도 범위내에서 임의로 원본이미지를 변형
    fill_mode='nearest',    # 빈자리를 채워준다
    validation_split = 0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255, # 원래 값 0 ~ 255 -> rescale 후 0 ~ 1
    preprocessing_function= preprocess_input
) 

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../study/LPD_COMPETITION/train', 
    target_size=(64, 64),
    batch_size=48000,                              # 출력되는 y값 개수 설정
    class_mode='categorical',
    subset = "training"
)

xy_test = train_datagen.flow_from_directory(
    '../study/LPD_COMPETITION/train',
    target_size=(64, 64),
    batch_size=48000,
    class_mode='categorical',
    subset = "validation"
)

print(xy_train[0][0].shape)  # (39000, 64, 64, 3) 
print(xy_train[0][1].shape)  # (39000, 1000)
print(xy_test[0][0].shape)  # (9000, 64, 64, 3)
print(xy_test[0][1].shape)  # (9000, 1000) 

np.save('../study/LPD_COMPETITION/npy/train_x.npy', arr=xy_train[0][0])
np.save('../study/LPD_COMPETITION/npy/train_y.npy', arr=xy_train[0][1])
np.save('../study/LPD_COMPETITION/npy/test_x.npy', arr=xy_test[0][0])
np.save('../study/LPD_COMPETITION/npy/test_y.npy', arr=xy_test[0][1])


