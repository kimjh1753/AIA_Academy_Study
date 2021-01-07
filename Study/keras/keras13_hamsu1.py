import numpy as np
# 1. 데이터
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
# from keras.layers import Dense

input1 = Input(shape=(5,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(2)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# model = Sequential()
# model.add(Dense(5, activation='relu', input_shape=(5,)))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(2))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=5000, batch_size=1, 
          validation_split=0.2, verbose=3)

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
# print("mse : ", mean_squared_error(y_predict, y_test))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


y_pred2 = model.predict(x_pred2)
print(y_pred2)

# [[187.35754  83.19966]]
