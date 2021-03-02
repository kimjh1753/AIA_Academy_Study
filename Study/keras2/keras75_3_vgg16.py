from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) 
                                # include_top=False으로 설정해야 내가 원하는 shape로 구성 가능
# print(model.weights)

vgg16.trainable = False
vgg16.summary()
print(len(vgg16.weights))           # 26 -> layer 하나당 (weights, bias)해서 2개씩 13개의 layer로 구성            
print(len(vgg16.trainable_weights)) # 0  -> model.trainable = False 이라서 구성 안함

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) #, activation='softmax'))
model.summary()
print("그냥 가중치의 수 : ", len(model.weights))           
# 32 -> layer 하나당 (weights, bias)해서 2개씩 13개의 layer로 구성 + 새로 구성한 모델 layer 하나당 (weights, bias)해서 2개씩 3개의 layer로 구성            
print("동결한 후 훈련되는 가중치의 수 : ", len(model.trainable_weights)) 
# 6  -> 새로 구성한 모델 layer 하나당 (weights, bias)해서 2개씩 3개의 layer로 구성

'''
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 32, 32, 64)        1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 32, 32, 64)        36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 16, 16, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 16, 16, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 16, 16, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 8, 8, 128)         0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 8, 8, 256)         295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 8, 8, 256)         590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 8, 8, 256)         590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 4, 4, 256)         0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 4, 4, 512)         1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 4, 4, 512)         2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 4, 4, 512)         2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 2, 2, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 2, 2, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 2, 2, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 2, 2, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 1, 1, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
_________________________________________________________________
26
0
'''

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 1, 1, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 10)                5130
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 14,719,879
Trainable params: 5,191
Non-trainable params: 14,714,688
_________________________________________________________________
32
6
'''

########################## 요기 하단 때문에 파일 분리했다.

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)
'''
                                                                            Layer Type Layer Name  Layer Trainable
0  <tensorflow.python.keras.engine.functional.Functional object at 0x0000019E3E9782E0>  vgg16      False
1  <tensorflow.python.keras.layers.core.Flatten object at 0x0000019E3E984880>           flatten    True
2  <tensorflow.python.keras.layers.core.Dense object at 0x0000019E3E9A66A0>             dense      True
3  <tensorflow.python.keras.layers.core.Dense object at 0x0000019E3E9AA6D0>             dense_1    True
4  <tensorflow.python.keras.layers.core.Dense object at 0x0000019E3E9CAD90>             dense_2    True
'''