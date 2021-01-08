# 실습 
# 드랍아웃 적용
# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성하시오.

# 1. 데이터
import numpy as np
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (442, 10) (442,)
print(x[:5])
print(y[:10])
 
print(np.max(x), np.min(x)) # 0.198787989657293 -0.137767225690012
print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

model= Sequential()
model.add(Dense(200, activation='relu', input_shape=(10,)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, batch_size=8, epochs=2000, 
          validation_split=0.2, verbose=1, callbacks=[es])

# 4. 평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 실습 1
# loss, mae :  3393.61083984375 47.12013244628906
# RMSE :  58.2547106883262
# mse :  3393.611317380587
# R2 :  0.47710474684639814

# Dropout 이후
# loss, mae :  5548.6689453125 54.8373908996582
# RMSE :  74.48939624215046
# mse :  5548.670152520099
# R2 :  0.14504844169741204
# 노드의 수가 부족했던거 같다.


