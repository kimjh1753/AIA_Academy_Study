import os, cv2, glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import pyramid_reduce

# 경로
base_path = r'C:\project\celeba-dataset\processed2'   
img_base_path = os.path.join(base_path, 'img') 
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

h, w, _ = img_sample.shape

#이미지 전처리

# 정사각형 이미지로 crop 해준다.
crop_sample = img_sample[int((h-w)/2):int(-(h-w)/2), :]

# 이미지를 4배만큼 축소하고 normalize 한다.
resized_sample = pyramid_reduce(crop_sample, 
                                downscale=4,
                                multichannel=True) # multichannel=True -> 컬러채널 허용

print(crop_sample.shape) # (178, 178, 3)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_sample)
plt.subplot(1, 3, 2)
plt.imshow(crop_sample)
plt.subplot(1, 3, 3)
plt.imshow(resized_sample)
plt.show()

# main
downscale = 4
n_train = 18650
n_val = 5700
n_test = 5649

for i, e in enumerate(eval_list):
    filename, ext = os.path.splitext(e[0])
    
    img_path = os.path.join(img_base_path, e[0])
    
    img = cv2.imread(img_path)
    
    h, w, _ = img.shape
    
    crop = img[int((h-w)/2):int(-(h-w)/2), :]
    crop = cv2.resize(crop, dsize=(176,176))
    resized = pyramid_reduce(crop, downscale=downscale, multichannel=True) # multichannel=True -> 컬러채널 허용
    
    norm = cv2.normalize(crop.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)
    
    if int(e[1]) == 0: # Train
        np.save(os.path.join(target_train_img_path, 'x_train', filename + '.npy'), resized)
        np.save(os.path.join(target_train_img_path, 'y_train', filename + '.npy'), norm)
    elif int(e[1]) == 1: # Validation
        np.save(os.path.join(target_val_img_path, 'x_val', filename + '.npy'), resized)
        np.save(os.path.join(target_val_img_path, 'y_val', filename + '.npy'), norm)
    elif int(e[1]) == 2: # Test
        np.save(os.path.join(target_test_img_path, 'x_test', filename + '.npy'), resized)
        np.save(os.path.join(target_test_img_path, 'y_test', filename + '.npy'), norm)   
    break

x_train_path = '../project/celeba-dataset/processed/x_train/'
second_path = '.npy'
train_list = []

for i in range(1, 80270):
    i = '{0:06d}'.format(i)
    train_path = x_train_path + i + second_path
    # print(x_train_path.format(train_path))
    a = np.load(train_path)
    train_list.append(a)
x_tarin = np.array(train_list).reshape(-1,44,44,1)
train_list = np.save(x_train_path, arr=train_list)

x_train = np.load('../project/celeba-dataset/processed/x_train/a.npy')
print(x_train.shape)





