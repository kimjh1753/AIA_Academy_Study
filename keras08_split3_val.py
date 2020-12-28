from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1, 101))
y = np.array(range(1, 101))

# x_train = x[:60]    # 순서 0번째부터 59번째까지 :::: 값 1 ~ 60 
# x_val = x[60:80]    # 61 ~ 80
# x_test = x[80:]     # 81 ~ 100
# 리스트의 슬라이싱 

# y_train = y[:60]    # 순서 0번째부터 59번째까지 :::: 값 1 ~ 60 
# y_val = y[60:80]    # 61 ~ 80
# y_test = y[80:]     # 81 ~ 100
# 리스트의 슬라이싱 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
print(y_predict)

# Shuffle이 False일때
# loss :  0.02788078784942627
# mae :  0.16663742065429688

# Shuffle이 True일때
# loss :  1.9114318661195284e-07
# mae :  0.00036057233228348196

# validation = 0.2
# loss :  0.002567283809185028
# mae :  0.041400570422410965