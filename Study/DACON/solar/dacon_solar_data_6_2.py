import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.backend import mean, maximum
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train = pd.read_csv('../study/dacon/data/train/train.csv')

def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

# 예측할 Target 칼럼 추가하기
def preprocess_data (data, is_train=True) :
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    
    if is_train == True :    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날 TARGET을 붙인다.
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날 TARGET을 붙인다.
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 이틀치 데이터만 빼고 전체
    
    elif is_train == False :         
        # Day, Minute 컬럼 제거
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:, :] # 마지막 하루치 데이터

df_train = preprocess_data(train)
print(df_train.shape) # (52464, 9)

# test 데이터 불러오기 >> x_pred
df_test = []

for i in range(81):
    file_path = '../STUDY/DACON/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

df_test = pd.concat(df_test)
x_predict = df_test.to_numpy()

# x_train을 RNN식으로 데이터 자르기
aaa = df_train.values

def split_xy(aaa, x_row, x_col, y_row, y_col):
    x, y = list(), list()
    for i in range(len(aaa)):
        if i > len(aaa)-x_row:
            break
        tmp_x = aaa[i:i+x_row, :x_col]
        tmp_y = aaa[i:i+x_row, x_col:x_col+y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

# print(x, '\n\n', y)
x_train, y_train = split_xy(aaa, 48, 7, 48, 2)     # 30분씩 RNN식으로 자름
print(x_train.shape)    #(52417, 48, 7)
print(y_train.shape)    #(52417, 48, 2)

x_predict = x_predict.reshape(81, 48, 7)
x_predict = x_predict.reshape(81, 48*7)

#===================================================================
# 데이터 전처리 : 준비 된 데이터 x_train / y_train / all_test
# 1) 트레인테스트분리 / 2) 민맥스or스탠다드 / 3) 모델에 넣을 쉐잎

# 1) 2차원으로 만들어서 트레인테스트분리
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

# 2) 민맥스or스탠다드
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)

print(x_train.shape, y_train.shape) # (33546, 336) (33546, 96)
print(x_val.shape, y_val.shape) # (8387, 336) (8387, 96)
print(x_test.shape, y_test.shape) # (10484, 336) (10484, 336)
print(x_predict.shape) # (81, 336)

x_train = x_train.reshape(x_train.shape[0], 1, 48, 7)
x_val = x_val.reshape(x_val.shape[0], 1, 48, 7)
x_test = x_test.reshape(x_test.shape[0], 1, 48, 7)
x_predict = x_predict.reshape(x_predict.shape[0], 1, 48, 7)

y_train = y_train.reshape(y_train.shape[0], 48, 2)
y_test = y_test.reshape(y_test.shape[0], 48, 2)
y_val = y_val.reshape(y_val.shape[0], 48, 2)

print(x_train.shape, y_train.shape) # (33546, 1, 48, 7) (33546, 48, 2)
print(x_val.shape, y_val.shape) # (8387, 1, 48, 7) (8387, 48, 2)
print(x_test.shape, y_test.shape) # (10484, 1, 48, 7) (10484, 48, 2)
print(x_predict.shape) # (81, 1, 48, 7)

# tensorflow pinball loss, Quantile loss definition
from tensorflow.keras.backend import mean, maximum
def quantile_loss(q, y, pred):
    err = (y - pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)
# mean = 평균
# K 를 tensorflow의 백앤드에서 불러왔는데 텐서형식의 mean을 쓰겠다는 것이다.

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sub = pd.read_csv('../study/dacon/data/sample_submission.csv')

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Reshape

def mymodel():
    model = Sequential()
    model.add(Conv2D(filters=256, kernel_size=(1,2), padding='same', input_shape=(1, 48, 7)))
    model.add(Conv2D(filters=256, kernel_size=(1,2), activation = 'relu'))
    model.add(Conv2D(filters=256, kernel_size=(1,2), activation = 'relu'))
    model.add(Conv2D(filters=256, kernel_size=(1,2), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(96, activation= 'relu'))
    model.add(Reshape((48, 2)))
    model.add(Dense(2))
    return model

# 3. 컴파일, 훈련
for q in q_lst:
    print(str(q)+'번째 훈련')
    model = mymodel()
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(q, y, pred)])
    # modelpath = '../data/modelcheckpoint/dacon_solar_01_{epoch:02d}-{val_loss:.4f}.hdf5'    # 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
    # cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.2, verbose=1, mode='auto')
    hist = model.fit(x_train, y_train, epochs=10000, batch_size=64, callbacks=[early_stopping, reduce_lr], validation_data=(x_val, y_val))

    # 평가, 예측
    result = model.evaluate(x_test, y_test, batch_size=48)
    print('loss : ', result[0])
    print('mae : ', result[1])
    y_predict = model.predict(x_predict)
    print(y_predict.shape) # (81, 48, 2)

    # 예측값을 submission에 넣기
    y_predict = pd.DataFrame(y_predict.reshape(y_predict.shape[0]*48, 2))
    y_predict2 = pd.concat([y_predict], axis=1)
    y_predict2[y_predict < 0] = 0
    y_predict3 = y_predict2.to_numpy()

    print(str(q)+'번째 지정')
    sub.loc[sub.id.str.contains('Day7'), 'q_' + str(q)] = y_predict3[:,0].round(2)
    sub.loc[sub.id.str.contains('Day8'), 'q_' + str(q)] = y_predict3[:,1].round(2)
    
sub.to_csv('../STUDY/DACON/data/submission_0120_6_2.csv', index=False)

