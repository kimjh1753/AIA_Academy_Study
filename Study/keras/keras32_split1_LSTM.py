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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4,1)))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

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

# keras32_split1_LSTM
# 0.0034312570933252573
# [[11.162672]]

