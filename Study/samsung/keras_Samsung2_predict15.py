import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

x_train = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[0]
x_test = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[1]
x_val = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[2]
y_train = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[3]
y_test = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[4]
y_val = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[5]
x_pred = np.load('./samsung/samsung2_data15.npy', allow_pickle=True)[6]
print(x_train.shape, x_test.shape, x_val.shape) # (1530, 6, 6) (479, 6, 6) (383, 6, 6)
print(y_train.shape, y_test.shape, y_val.shape) # (1530, 1) (479, 1) (383, 1)
print(x_pred.shape) # (1, 6, 6)

# 2~3. 모델 구성, (컴파일, 훈련)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten

# 4. 평가, 예측
model = load_model('./samsung/keras_Samsung0115.h5')
result = model.evaluate(x_test, y_test)
print("로드체크포인트_loss : ", result[0])
print("로드체크포인트_accuracy : ", result[1])

y_predict = model.predict(x_test)

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

predict = model.predict(x_pred)
print("1월 15일 삼성주가 예측 : ", predict)
