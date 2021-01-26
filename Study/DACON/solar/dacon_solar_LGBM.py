import numpy as np
import pandas as pd
import os
import glob
import random
import tensorflow.keras.backend as K
from tensorflow.keras.backend import mean, maximum

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('../study/dacon/data/train/train.csv')
print(train.shape) # (52560, 9)
print(train.tail())
submission = pd.read_csv('../study/dacon/data/sample_submission.csv')
print(submission.shape)
submission.tail() # (7776, 10)

#1. DATA

# train data column 정리
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

# 예측할 Target 칼럼 추가하기(끝에 다음날, 다다음날 TARGET 데이터를 column을 추가한다.)
def preprocess_data (data, is_train=True) :
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    
    if is_train == True :    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날 TARGET을 붙인다.
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날 TARGET을 붙인다.
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 이틀치 데이터만 빼고 전체 (예측 해야하는 날짜라서)
    
    elif is_train == False :         
        # Day, Minute 컬럼 제거
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:, :] # 0 ~ 6일 중 마지막 6일 데이터만 남긴다. (6일 데이터로 7, 8일을 예측하고자 함)

df_train = preprocess_data(train)
print(df_train.shape) # (52464, 9)
print(df_train.iloc[:48])
print(df_train.iloc[48:96])
print(df_train.iloc[48+48:96+48])
print(df_train.tail())

# test 데이터 불러오기(81개의 0 ~ 7 Day 데이터 합치기) >> x_pred
df_test = []

for i in range(81):
    file_path = '../STUDY/DACON/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_test = pd.concat(df_test)
print(x_test.shape) # (3888, 7) -> 27216
print(x_test.head(48))

print(df_train.head())
print(df_train.iloc[-48:])

from sklearn.model_selection import train_test_split
x_train1, x_val1, y_train1, y_val1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], train_size=0.8, random_state=66)
x_train2, x_val2, y_train2, y_val2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], train_size=0.8, random_state=66)

print(x_train1.head())
print(x_test.head())

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train1)
x_train1 = scaler.transform(x_train1)
x_val1 = scaler.transform(x_val1)
x_train2 = scaler.transform(x_train2)
x_val2 = scaler.transform(x_val2)
x_test = scaler.transform(x_test)

print(x_train1.shape, x_val1.shape) # (41971, 7) (10493, 7)
print(x_train2.shape, x_val2.shape) # (41971, 7) (10493, 7)
print(y_train1.shape, y_val1.shape) # (41971,) (10493,)
print(y_train2.shape, y_val2.shape) # (41971,) (10493,)
print(x_test.shape) # (3888, 7)

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

from lightgbm import LGBMRegressor

# 2 . 모델 구성 3. 컴파일 훈련 4. 평가, 예측
# Get the model and the predictions in (a) - (b)
def LGBM(q, x_train, y_train, x_val, y_val, x_test):

    # (a) Modeling
    model = LGBMRegressor(objective='quantile', alpha=q, max_depth=-1,
                          n_estimators=10000, bagging_fraction=0.7, learning_rate=0.01, subsample=0.7)
    # objective = 'quantile' >> quantile 회귀모델
    # n_estimators           >> (default=100) 훈련시킬 tree의 개수
    # bagging_fraction       >> 0 ~ 1 사이, 랜덤 샘플링
    # learning_rate          >> 일반적으로 0.01 ~ 0.1 사이
    # subsample              >> Row sampling, 즉 데이터를 일부 발췌해서 다양성을 높이는 방법으로 쓴다.
    # print("model : ", model)
    # 출력결과 >> model :  LGBMRegressor(alpha=0.5, bagging_fraction=0.7, learning_rate=0.027,
            #   n_estimators=10000, objective='quantile', subsample=0.7)

    model.fit(x_train, y_train, eval_metric=['quantile'],
              eval_set=[(x_val, y_val)], early_stopping_rounds=300, verbose=500)
    # early_stopping_rounds  >> validation셋에 더이상 발전이 없으면 그만두게 설정할때 이를 몇번동안 발전이 없으면 그만두게 할지 여부.

    # (b) Predictions
    pred = pd.Series(model.predict(x_test).round(2))
    # Series : 1차원 배열
    # print(pred) : X_test로 predict한 결과가 나옴
    return pred, model

# Target 예측

def train_data(x_train, y_train, x_val, y_val, x_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in q_lst:
        print(q)
        pred, model = LGBM(q, x_train, y_train, x_val, y_val, x_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred], axis=1)
        LGBM_actual_pred[LGBM_actual_pred < 0] = 0
    LGBM_actual_pred.columns = q_lst

    return LGBM_models, LGBM_actual_pred

# Target1
models_1, results_1 = train_data(x_train1, y_train1, x_val1, y_val1, x_test)
# results_1.sort_index()[:48]

# Target2
modells_2, results_2 = train_data(x_train2, y_train2, x_val2, y_val2, x_test)
results_2.sort_index()[:48]

print(results_1.shape, results_2.shape) # (3888, 9) (3888, 9)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values

submission.to_csv('../STUDY/DACON/data/sample_submission_0120_3.csv', index=False)


