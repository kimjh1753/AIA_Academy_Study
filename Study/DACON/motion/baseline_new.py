import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import tensorflow as tf

from keras.layers.advanced_activations import LeakyReLU, PReLU
from math import cos, sin, pi
from PIL import Image
from tqdm import tqdm
from tensorflow.keras import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

#폴더 경로를 설정해줍니다.
os.chdir('../study/dacon/data-3/1. open') 

root_dir = '../study/dacon/data-3/1. open'

os.makedirs(root_dir +'/train')
os.makedirs(root_dir +'/val')

# validation용 파일은 10% 비율로 random sampling
# random.seed() 넣으시면 복원 가능
src = "train_imgs"
all_filename = os.listdir(src)
valid_filename = random.sample(all_filename, int(len(train_all) * 0.1))
train_filename = [x for x in all_filename if x not in valid_filename]

print(len(train_filename), len(valid_filename))

train_filename = [src+'/'+ name for name in train_filename]
valid_filename = [src+'/' + name for name in valid_filename]

print('Total images: ', len(all_filename))
print('Training: ', len(train_filename))
print('Validation: ', len(valid_filename))

# copy & paste images
for name in tqdm(train_filename):
    shutil.copy(name, 'train')

for name in tqdm(valid_filename):
    shutil.copy(name, 'val')

