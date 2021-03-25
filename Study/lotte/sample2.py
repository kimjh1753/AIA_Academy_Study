import numpy as np
import PIL
from numpy import asarray
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
import string
import scipy.signal as signal
from keras.applications.resnet import ResNet101,preprocess_input

# img1=[]
# for i in range(0,72000):
#     filepath='../study/LPD_COMPETITION/test/%d.jpg'%i
#     image2=Image.open(filepath)
#     image2 = image2.convert('RGB')
#     image2 = image2.resize((72, 72))
#     image_data2=asarray(image2)
#     # image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
#     img1.append(image_data2)    

# np.save('../study/LPD_COMPETITION/npy/pred.npy', arr=img1)

x_pred = np.load('../study/LPD_COMPETITION/npy/pred.npy',allow_pickle=True)

print(x_pred.shape)