import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.transform import pyramid_reduce

base_path = r'C:\project\celeba-dataset/processed2'   
img_base_path = os.path.join(base_path, 'img_align_celeba') 
target_train_img_path = os.path.join(base_path, 'train')
target_test_img_path = os.path.join(base_path, 'test')
target_val_img_path = os.path.join(base_path, 'val')

eval_list = np.loadtxt(os.path.join(base_path, 'list_eval_partition2.csv'), 
                       dtype=str, 
                       delimiter=',', 
                       skiprows=1)

print(eval_list[0]) # ['000001.jpg' '0']

# 이미지 확인
img_sample = cv2.imread(os.path.join(img_base_path, 
                                     eval_list[0][0]))


plt.imshow(img_sample)
plt.show()

h, w, _ = img_sample.shape

#이미지 전처리

# 정사각형 이미지로 crop 해준다.
crop_sample = img_sample[int((h-w)/2):int(-(h-w)/2), :]

# 이미지를 4배만큼 축소하고 normalize 한다.
resized_sample = pyramid_reduce(crop_sample, 
                                downscale=4,
                                multichannel=True) # multichannel=True -> 컬러채널 허용

pad = int((crop_sample.shape[0] - resized_sample.shape[0]) / 2)

padded_sample = cv2.copyMakeBorder(resized_sample, 
                                   top=pad, 
                                   bottom=pad, 
                                   left=pad, 
                                   right=pad, 
                                   borderType=cv2.BORDER_CONSTANT, 
                                   value=(0,0,0))

plt.imshow(padded_sample)
plt.show()

cv2.imwrite('../Project/1.jpg', padded_sample)
