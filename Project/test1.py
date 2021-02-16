import numpy as np
import cv2

x = np.load('../Project/celeba-dataset/processed2/train/x_train/000001.npy')
print(x)


cv2.imshow(x)
cv2.show()