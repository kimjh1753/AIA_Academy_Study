from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1

# model = VGG16()
# model = VGG19()
# model = Xception()
# model = ResNet101()
# model = ResNet101V2()
# model = ResNet152()
# model = ResNet152V2()
# model = ResNet50()
# model = ResNet50V2()
# model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# model = MobileNetV2()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
# model = NASNetLarge()
# model = NASNetMobile()
# model = EfficientNetB0()
model = EfficientNetB1()

model.trainable = False
# model.trainable = True

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# model = VGG16
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544
# __________________________________________________________________________________________________
# 32
# 0

# model = VGG19()
# Total params: 143,667,240
# Trainable params: 0
# Non-trainable params: 143,667,240
# __________________________________________________________________________________________________
# 38
# 0

# model = Xception()
# Total params: 22,910,480
# Trainable params: 0
# Non-trainable params: 22,910,480
# __________________________________________________________________________________________________
# 236
# 0

# model = ResNet101()
# Total params: 44,707,176
# Trainable params: 0
# Non-trainable params: 44,707,176
# __________________________________________________________________________________________________
# 626
# 0

# model = ResNet101V2()
# Total params: 44,675,560
# Trainable params: 0
# Non-trainable params: 44,675,560
# __________________________________________________________________________________________________
# 544
# 0

# model = ResNet152()
# Total params: 60,419,944
# Trainable params: 0
# Non-trainable params: 60,419,944
# __________________________________________________________________________________________________
# 932
# 0

# model = ResNet152V2()
# Total params: 60,380,648
# Trainable params: 0
# Non-trainable params: 60,380,648
# __________________________________________________________________________________________________
# 816
# 0

# model = ResNet50()
# Total params: 25,636,712
# Trainable params: 0
# Non-trainable params: 25,636,712
# __________________________________________________________________________________________________
# 320
# 0

# model = ResNet50V2
# Total params: 25,613,800
# Trainable params: 0
# Non-trainable params: 25,613,800
# __________________________________________________________________________________________________
# 272
# 0

# model = InceptionV3()
# Total params: 23,851,784
# Trainable params: 0
# Non-trainable params: 23,851,784
# __________________________________________________________________________________________________
# 378
# 0

# model = InceptionResNetV2()
# Total params: 55,873,736
# Trainable params: 0
# Non-trainable params: 55,873,736
# __________________________________________________________________________________________________
# 898
# 0

# model = MobileNet()
# Total params: 4,253,864
# Trainable params: 0
# Non-trainable params: 4,253,864
# _________________________________________________________________
# 137
# 0

# model = MobileNetV2()
# Total params: 3,538,984
# Trainable params: 0
# Non-trainable params: 3,538,984
# __________________________________________________________________________________________________
# 262
# 0

# model = DenseNet121()
# Total params: 8,062,504
# Trainable params: 0
# Non-trainable params: 8,062,504
# __________________________________________________________________________________________________
# 606
# 0

# model = DenseNet169()
# Total params: 14,307,880
# Trainable params: 0
# Non-trainable params: 14,307,880
# __________________________________________________________________________________________________
# 846
# 0

# model = DenseNet201()
# Total params: 20,242,984
# Trainable params: 0
# Non-trainable params: 20,242,984
# __________________________________________________________________________________________________
# 1006
# 0

# model = NASNetLarge()
# Total params: 88,949,818
# Trainable params: 0
# Non-trainable params: 88,949,818
# __________________________________________________________________________________________________
# 1546
# 0

# model = NASNetMobile()
# Total params: 5,326,716
# Trainable params: 0
# Non-trainable params: 5,326,716
# __________________________________________________________________________________________________
# 1126
# 0

# model = EfficientNetB0()
# Total params: 5,330,571
# Trainable params: 0
# Non-trainable params: 5,330,571
# __________________________________________________________________________________________________
# 314
# 0

# model = EfficientNetB1()
# Total params: 7,856,239
# Trainable params: 0
# Non-trainable params: 7,856,239
# __________________________________________________________________________________________________
# 442
# 0


