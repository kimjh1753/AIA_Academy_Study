# [실습] keras67_1 남자 여자에 noise를 넣어서
# 기미 주근깨 여드름을 제거하시오

import numpy as np

x_train = np.load('../data2/image/npy/keras67_train_x.npy')
x_test = np.load('../data2/image/npy/keras67_test_x.npy')

# x_train = x_train/255.
# x_test = x_test/255.

print(x_train.shape, x_test.shape) # (1390, 128, 128, 3) (347, 128, 128, 3)

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Conv2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(128, (2, 2), 1, padding = 'same', input_shape = (128, 128, 3)))
    model.add(Conv2D(32, (2, 2), 1, padding = 'same', activation = 'relu'))
    model.add(Dense(units = hidden_layer_size, activation = 'relu'))
    model.add(Conv2DTranspose(32, (2, 2), 1, padding = 'same', activation = 'relu'))
    model.add(Conv2DTranspose(128, (2, 2), 1, padding = 'same', activation = 'relu'))
    model.add(Dense(units = 3, activation = 'sigmoid'))
    return model

model = autoencoder(hidden_layer_size=30) # hidden_layer_size=154 : 95%의 pca 수치
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', patience=15, verbose=1, factor=0.5, mode='auto')

model.fit(x_train_noised, x_train, epochs=1000, validation_split=0.2, callbacks=[es, rl])
model.save('../study/AE/model/a08.h5')

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize=(20, 7))          

# 이미지 다섯개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다!!
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(128, 128, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지     
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(128, 128, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(128, 128, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])    

plt.tight_layout()
plt.show()