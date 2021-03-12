from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 1000, test_split = 0.2 
  # num_words = 불러올 단어수 설정
)

print(x_train[0], type(x_train[0]))
'''
[1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 
 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 
 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12] <class 'list'>
'''
print(y_train[0]) # 3
print(len(x_train[0]), len(x_train[1])) # 87 56
print("==============================")
print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)

print("뉴스기사 최대길이 : ", max(len(I) for I in x_train))             # 뉴스 기사 최대길이 :  2376
print("뉴스기사 평균길이 : ", sum(map(len, x_train)) / len(x_train))    # 뉴스기사 평균길이 :  145.5398574927633

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# y분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("y분포 : ", dict(zip(unique_elements, counts_elements)))
'''
y분포 :  {0: 55, 1: 432, 2: 74, 3: 3159, 4: 1949, 5: 17, 6: 48, 7: 16, 8: 139, 9: 101, 10: 124, 
         11: 390, 12: 49, 13: 172, 14: 26, 15: 20, 16: 444, 17: 39, 18: 66, 19: 549, 20: 269, 
         21: 100, 22: 15, 23: 41, 24: 62, 25: 92, 26: 24, 27: 15, 28: 48, 29: 19, 30: 45, 
         31: 39, 32: 32, 33: 11, 34: 50, 35: 10, 36: 49, 37: 19, 38: 19, 39: 24, 40: 36, 
         41: 30, 42: 13, 43: 21, 44: 12, 45: 18}
'''         
print("==============================")

# plt.hist(y_train, bins=46)
# plt.show()

# x의 단어를 분포
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index)) # <class 'dict'>
print("==============================")

# 키와 벨류를 교체!!!
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

# 키 벨류 교환후
print(index_to_word)
print(index_to_word[1])     # the
print(index_to_word[30979]) # northerly
print(len(index_to_word))   # 30979

# x_train[0]
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))

# y 카테고리 갯수 출력
category = np.max(y_train) + 1
print("y 카테고리 개수 : ", category) # y 카테고리 개수 :  46

# y의 유니크한 값 출력
y_bunpo = np.unique(y_train) 
print(y_bunpo)

####################################### 전처리 ##############################################################

# 1. 데이터
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

print(x_train.shape, x_test.shape) # (8982, 100) (2246, 100)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (8982, 46) (2246, 46)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Embedding
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=120))
model.add(LSTM(120))
model.add(Dense(46, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=4, verbose=1)
rl = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2, verbose=1)

model.fit(x_train, y_train, batch_size=64, epochs=1000, validation_split=0.2, callbacks=[es, rl])

result = model.evaluate(x_test, y_test)
print("loss : ", result[0]) 
print("acc : ", result[1])

# epochs = 30
# loss :  1.2671891450881958
# acc :  0.7101513743400574




