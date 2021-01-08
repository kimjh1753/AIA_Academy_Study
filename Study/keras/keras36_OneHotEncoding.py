import numpy as np

# 1. 데이터

x = np.array([3, 6, 5, 4, 2])
print(x.shape)  # (5,) -> (5, 7)

# OneHotEncoding(tensorflow)
# from tensorflow.keras.utils import to_categorical

# x = to_categorical(x)

# print("x 데이터에 대한 tensorflow to_categorical ")
# print(x)
# print(x.shape)


# x 데이터에 대한 tensorflow to_categorical 
# [[0. 0. 0. 1. 0. 0. 0.]       3번째
#  [0. 0. 0. 0. 0. 0. 1.]       6번째
#  [0. 0. 0. 0. 0. 1. 0.]       5번째
#  [0. 0. 0. 0. 1. 0. 0.]       5번째
#  [0. 0. 1. 0. 0. 0. 0.]]      2번째
# (5, 7)


# OneHotEncoding(sklearn)
# 리스트 나열 기준은 영어 > 한글 순이고, 내림차순으로 정렬
from sklearn.preprocessing import OneHotEncoder
x = x.reshape(-1, 1)

one = OneHotEncoder()
one.fit(x)
x = one.transform(x).toarray()

print("x 데이터에 대한 sklearn OneHotEndcoding ")
print(x)
print(x.shape) # (5,) -> (5,5)

# x 데이터에 대한 sklearn OneHotEndcoding 
#   2  3  4  5  6   ->  3, 6, 5, 4, 2를 내림차순 후 각 위치에 1 삽입
# [[0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 1. 0. 0.]
#  [1. 0. 0. 0. 0.]]
# (5, 5)