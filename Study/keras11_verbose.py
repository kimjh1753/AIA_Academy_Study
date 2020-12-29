import numpy as np
#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101), 
              range(701, 801), range(501, 601)])
y = np.array([range(711, 811), range(1,101)])
print(x.shape)      # (5, 100)
print(y.shape)      # (2, 100)
x_pred2 = np.array([100, 402, 101, 100, 401])
print("x_pred2.shape : ", x_pred2.shape)     # (5, )       


# x = np.arange(20).reshape(10,2)
x = np.transpose(x)
y = np.transpose(y)
# x_pred2 = np.transpose(x_pred2)
x_pred2 = x_pred2.reshape(1, 5)

print(x)
print(x.shape)      # (100, 5)
print(y.shape)      # (100, 2)
print("x_pred2.shape : ", x_pred2.shape)    #(1, 5) 


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=66, shuffle=True
)
print(x_train.shape)        # (80, 5)
print(y_train.shape)        # (80, 2)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, 
          validation_split=0.2, verbose=1)

"""
verbose = 0 : 
epoch 값 아예 안나옴
1/1 [==============================] - 0s 1ms/step - loss: 1.5817e-07 - mae: 3.0160e-04
loss :  1.5816880249985843e-07
mae :  0.0003015950205735862
RMSE :  0.00039770441699737134
R2 :  0.9999999997998692
[[494.2784  419.09204]]

verbose = 1 :
Epoch 100/100
64/64 [==============================] - 0s 1ms/step - loss: 4.4761e-09 - mae: 5.0228e-05 - val_loss: 2.9777e-09 - val_mae: 4.2200e-05
1/1 [==============================] - 0s 985us/step - loss: 2.1045e-09 - mae: 3.5934e-05
loss :  2.104534546631953e-09
mae :  3.593415021896362e-05
RMSE :  4.5875206713876224e-05
R2 :  0.9999999999973371
[[523.58704 106.38356]]

verbose = 2 :
Epoch 100/100
64/64 - 0s - loss: 3.4483e-08 - mae: 1.5508e-04 - val_loss: 1.3747e-08 - val_mae: 9.4295e-05
1/1 [==============================] - 0s 994us/step - loss: 1.2835e-08 - mae: 9.0706e-05
loss :  1.2835227103380475e-08
mae :  9.07063513295725e-05
RMSE :  0.00011329266051137113
R2 :  0.9999999999837597
[[337.05075  -38.176983]]

verbose = 3 :
Epoch 100/100
1/1 [==============================] - 0s 2ms/step - loss: 1.4159e-09 - mae: 2.8095e-05
loss :  1.4159301509053535e-09
mae :  2.8094649678678252e-05
RMSE :  3.7628847028588226e-05
R2 :  0.9999999999982084
[[392.683   196.48792]]

"""

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


y_pred2 = model.predict(x_pred2)
print(y_pred2)

# [[187.35754  83.19966]]