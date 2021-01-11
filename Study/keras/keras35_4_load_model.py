# 35_3을 카피
import numpy as np

# 1. 데이터
a = np.array(range(1, 11))
size = 5

# 모델을 구성하시오.

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)    

dataset = split_x(a, size)
print("==========================")
print(dataset)  

x = dataset[:,0:4] # 0열부터 3열까지 자른 값을 보여준다.
y = dataset[:,4:] # 4열부터 끝까지 자른다.
print(x)
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]
#  [5 6 7]
#  [6 7 8]]
print(y)
# [[ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]
print(x.shape)  # (6, 4)
print(y.shape)  # (6, 1)

x = x.reshape(6, 4, 1)

# 2. 모델 구성
from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')
# 요 밑 3줄 넣고 테스트 ##########
from tensorflow.keras.layers import Dense
model.add(Dense(5, name='kingkeras1'))   # 이름 : dense
model.add(Dense(1, name='kingkeras2'))   # 이름 : dense_1
#############################

model.summary()

# 완성해보시오

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, validation_split=0.2, verbose=1, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print(loss)

x_pred = np.array([7, 8, 9, 10])
x_pred = x_pred.reshape(1, 4, 1)
result = model.predict(x_pred)
print(result)

# 0.01448071002960205
# [[10.740932]]