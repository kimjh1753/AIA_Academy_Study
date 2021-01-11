import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (y_train, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_test.shape, y_test.shape)   # (10000, 1) (10000, 1)
print(y_train)

plt.imshow(x_train[0])
plt.show()

