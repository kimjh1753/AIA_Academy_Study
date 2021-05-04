import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Prepare Automodel for search
input_node = ak.Input()
output_node = ak.ConvBlock()(input_node)
output_node = ak.DenseBlock()(output_node) #optional
output_node = ak.SpatialReduction()(output_node) #optional
output_node = ak.ClassificationHead(num_classes=2, multi_label=True)(output_node)

auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=1, overwrite=True
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=2)
ck = ModelCheckpoint('./temp/', save_weights_only=True,
                     save_best_only=True, monitor='val_loss', verbose=1)

auto_model.fit(x_train, y_train, epochs=1, validation_split=0.2,
          callbacks=[es, lr, ck]
)

model = auto_model.export_model()
model.summary()

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0
_________________________________________________________________
cast_to_float32 (CastToFloat (None, 28, 28, 1)         0
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 32)        9248
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 32)        9248
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 32)          9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 4, 4, 32)          0
_________________________________________________________________
flatten (Flatten)            (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 32)                16416
_________________________________________________________________
re_lu (ReLU)                 (None, 32)                0
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056
_________________________________________________________________
re_lu_1 (ReLU)               (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                330
_________________________________________________________________
classification_head_1 (Activ (None, 10)                0
=================================================================
Total params: 45,866
Trainable params: 45,866
Non-trainable params: 0
_________________________________________________________________
'''

results = auto_model.evaluate(x_test, y_test)

print(results)

# [0.01690400019288063, 0.9760000109672546]
