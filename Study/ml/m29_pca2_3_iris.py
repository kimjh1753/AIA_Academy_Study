import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
from sklearn.decomposition import PCA

# dataset = load_diabetes()
dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (150, 4) (150,)

# pca = PCA(n_components=11)
# x2 = pca.fit_transform(x)
# print(x2)
# print(x2.shape)         # (442, 7)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR) 
# print(sum(pca_EVR))

# 7 : 0.9479436357350414
# 8 : 0.9913119559917797
# 9 : 0.9991439470098977
# 10 : 1.0

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) # cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
print("cumsum : ", cumsum) # cumsum :  [0.92461872 0.97768521 0.99478782 1.        ]

d = np.argmax(cumsum > 0.95)+1
print("cumsum >= 0.95", cumsum >= 0.95) # cumsum >= 0.95 [False  True  True  True]
print("d : ", d) # d :  2

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()