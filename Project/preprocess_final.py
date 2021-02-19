import os, cv2, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce

# 경로
base_path = r'C:\project\celeba-dataset'   
img_base_path = os.path.join(base_path, 'img_align_celeba2') 
target_img_path = os.path.join(base_path, 'processed')

eval_list = np.loadtxt(os.path.join(base_path, 'list_eval_partition.csv'), 
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

# 이미지를 4배만큼 축소하고 normalize(정규화) 한다. ex) downscale=4 : 원본 사진 대비 4배 축소됨
resized_sample = pyramid_reduce(crop_sample, 
                                downscale=4,
                                multichannel=True) # multichannel=True -> 컬러채널 허용

print(crop_sample.shape) # (178, 178, 3)

plt.figure(figsize=(12, 5)) # 최초 창의 크기를 가로 12인치 세로 5인치로 설정
plt.subplot(1, 3, 1)        # 1행 3열 중 첫번째
plt.imshow(img_sample)      # 기존 이미지
plt.subplot(1, 3, 2)        # 1행 3열 중 두번째
plt.imshow(crop_sample)     # 기존 이미지를 정사각형으로 자른 이미지
plt.subplot(1, 3, 3)        # 1행 3열 중 세번째
plt.imshow(resized_sample)  # crop 이미지를 4배 축소한 이미지
plt.show()

# main
downscale = 4
n_train = 18650
n_val = 5700
n_test = 5649

for i, e in enumerate(eval_list):
    filename, ext = os.path.splitext(e[0]) # os.path.splitext : 확장자만 따로 분류한다.(리스트로 나타낸다)
    
    img_path = os.path.join(img_base_path, e[0]) 
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) # img_path에 있는 있는 이미지를 부른다.
    
    h, w, _ = img.shape
    
    crop = img[int((h-w)/2):int(-(h-w)/2), :] # 정사각형 이미지로 crop 해준다.
    crop = cv2.resize(crop, dsize=(176,176)) # dsize=(176,176) : 결과 이미지 크기는 Tuple형을 사용하고 176, 176으로 변경
    # 이미지를 4배만큼 축소하고 normalize(정규화) 한다. ex) downscale=4 : 원본 사진 대비 4배 축소됨
    resized = pyramid_reduce(crop, downscale=downscale, multichannel=True) # multichannel=True -> 컬러채널 허용
    
    norm = cv2.normalize(crop.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX) # -> 이미지를 0과 1 사이로 normalize(정규화)함
     
    if int(e[1]) == 0: # Train
        np.save(os.path.join(target_img_path, 'x_train', filename + '.npy'), resized)
        np.save(os.path.join(target_img_path, 'y_train', filename + '.npy'), norm)
    elif int(e[1]) == 1: # Validation
        np.save(os.path.join(target_img_path, 'x_val', filename + '.npy'), resized)
        np.save(os.path.join(target_img_path, 'y_val', filename + '.npy'), norm)
    elif int(e[1]) == 2: # Test
        np.save(os.path.join(target_img_path, 'x_test', filename + '.npy'), resized)
        np.save(os.path.join(target_img_path, 'y_test', filename + '.npy'), norm)

# ============================================================================================================= 
    
# x_train_path = '../project/celeba-dataset/processed/x_train/'
# second_path = '.npy'
# train_list = []

# for i in range(1, 301):
#     i = '{0:06d}'.format(i)
#     train_path = x_train_path + i + second_path
#     # print(x_train_path.format(train_path))
#     a = np.load(train_path)
#     train_list.append(a)
# x_train = np.array(train_list).reshape(-1,44,44,3)
# np.save(x_train_path, arr=x_train)

# x_train = np.load('../project/celeba-dataset/processed/x_train/x_train.npy')
# print(x_train.shape) # (300, 44, 44, 3)

# ===================================================================

# y_train_path = '../project/celeba-dataset/processed/y_train/'
# second_path = '.npy'
# train_list = []

# for i in range(1, 301):
#     i = '{0:06d}'.format(i)
#     train_path = y_train_path + i + second_path
#     # print(x_train_path.format(train_path))
#     a = np.load(train_path)
#     train_list.append(a)
# y_train = np.array(train_list).reshape(-1,176,176,3)
# np.save(y_train_path, arr=y_train)

# y_train = np.load('../project/celeba-dataset/processed/y_train/y_train.npy')
# print(y_train.shape) # (300, 176, 176, 3)

# ===================================================================

# x_val_path = '../project/celeba-dataset/processed/x_val/'
# second_path = '.npy'
# train_list = []

# for i in range(18651, 18951):
#     i = '{0:06d}'.format(i)
#     train_path = x_val_path + i + second_path
#     # print(x_train_path.format(train_path))
#     a = np.load(train_path)
#     train_list.append(a)
# x_val = np.array(train_list).reshape(-1,44,44,3)
# np.save(x_val_path, arr=x_val)

# x_val = np.load('../project/celeba-dataset/processed/x_val/x_val.npy')
# print(x_val.shape) # (300, 44, 44, 3)

# ===================================================================

# y_val_path = '../project/celeba-dataset/processed/y_val/'
# second_path = '.npy'
# train_list = []

# for i in range(18651, 18951):
#     i = '{0:06d}'.format(i)
#     train_path = y_val_path + i + second_path
#     # print(x_train_path.format(train_path))
#     a = np.load(train_path)
#     train_list.append(a)
# y_val = np.array(train_list).reshape(-1,176,176,3)
# np.save(y_val_path, arr=y_val)

# y_val = np.load('../project/celeba-dataset/processed/y_val/y_val.npy')
# print(y_val.shape) # (300, 176, 176, 3)

# ===================================================================

# x_test_path = '../project/celeba-dataset/processed/x_test/'
# second_path = '.npy'
# train_list = []

# for i in range(24351, 24651):
#     i = '{0:06d}'.format(i)
#     train_path = x_test_path + i + second_path
#     # print(x_train_path.format(train_path))
#     a = np.load(train_path)
#     train_list.append(a)
# x_test = np.array(train_list).reshape(-1,44,44,3)
# np.save(x_test_path, arr=x_test)

# x_test = np.load('../project/celeba-dataset/processed/x_test/x_test.npy')
# print(x_test.shape) # (300, 44, 44, 3)

# ===================================================================

# y_test_path = '../project/celeba-dataset/processed/y_test/'
# second_path = '.npy'
# train_list = []

# for i in range(24351, 24651):
#     i = '{0:06d}'.format(i)
#     train_path = y_test_path + i + second_path
#     # print(x_train_path.format(train_path))
#     a = np.load(train_path)
#     train_list.append(a)
# y_test = np.array(train_list).reshape(-1,176,176,3)
# np.save(y_test_path, arr=y_test)

# y_test = np.load('../project/celeba-dataset/processed/y_test/y_test.npy')
# print(y_test.shape) # (300, 176, 176, 3)