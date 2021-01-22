import numpy as np
import pandas as pd

################################

# 예측할 Target 칼럼 추가하기
def preprocess_data (data, is_train=True) :
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

# 시계열 데이터로 자르기
def split_xy(dataset, time_steps, y_row) :
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_row
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :-2]
        tmp_y = dataset[i:x_end_number, -2:]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

#1. DATA

# train 데이터 불러오기 >> x_train
train_pd = pd.read_csv('../STUDY/DACON/data/train/train.csv')
# print(train_pd.columns)    # Index(['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')
# print(train_pd.shape)      # (52560, 9)
df_train = preprocess_data(train_pd)
# print(df_train.columns) 
# Index(['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T', 'Target1', 'Target2'], dtype='object')
# print(df_train.shape)      # (52464, 9)

dataset = df_train.to_numpy()
# print(dataset.shape)      # (52464, 9)
# print(dataset[0])
# [  0.     0.     0.     0.     1.5   69.08 -12.     0.     0.  ]
x = dataset.reshape(-1, 48, 9)  # 하루치로 나눔
# print(x[0]) # day0

x, y = split_xy(dataset, 48 , 1)
# print(x.shape)     # (52416, 48, 7)  # day0 ~ day7, 7일씩 자름
# print(x[0:3])

# print(y.shape)     # (52416, 48, 2)
# print(y[0:2])  

# test 데이터 불러오기 >> x_pred
df_pred = []
for i in range(81):
    file_path = '../STUDY/DACON/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_pred.append(temp)

df_pred = pd.concat(df_pred)
# print(df_pred.shape) # (3888, 7) -> 27216
# print(df_pred.head())
pred_dataset = df_pred.to_numpy()

x_pred = pred_dataset.reshape(81, 48, 7)
# print(x_pred.shape) # (81, 48, 7)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape)    # (33545, 48, 7)
# print(x_test.shape)     # (10484, 48, 7)
# print(x_val.shape)      # (8387, 48, 7)

# print(y_train.shape)    # (33545, 48, 2)
# print(y_test.shape)     # (10484, 48, 2)
# print(y_val.shape)      # (8387, 48, 2)

x_train = x_train.reshape(x_train.shape[0]*x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0]*x_test.shape[1], x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0]*x_val.shape[1], x_val.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0]*x_pred.shape[1], x_pred.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(33545, 48, 7)
x_test = x_test.reshape(10484, 48, 7)
x_val = x_val.reshape(8387, 48, 7)
x_pred = x_pred.reshape(81, 48, 7)

y_train = y_train.reshape(33545, 48, 2)
y_test = y_test.reshape(10484, 48, 2)
y_val = y_val.reshape(8387, 48, 2)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, Reshape
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, padding='same', input_shape=(48, 7)))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Conv1D(filters=256, kernel_size=2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(48*2))
model.add(Reshape([48, 2]))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/dacon_solar_01_{epoch:02d}-{val_loss:.4f}.hdf5'
# 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기

early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, 
                 callbacks=[early_stopping, cp, reduce_lr], validation_data=(x_val, y_val))

model.save('./DACON/data/train/dacon_Solar0118.h5')

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("mae : ", result[1])

y_pred = model.predict(x_pred)
# print(y_pred)
# print(y_pred.shape) # (81, 48, 2)

y_pred = y_pred.reshape(3888, 2)
# print(y_pred)
# print(y_pred.shape) # (3888, 2)

sub = pd.read_csv("../STUDY/DACON/data/sample_submission.csv")

# 예측값을 제출 형식에 넣기 (예측한 값 9컬럼에 다 복붙함)
for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day7"), column_name] = y_pred[:,0]
for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day8"), column_name] = y_pred[:,1]

sub.to_csv('../STUDY/DACON/data/submission_data_3.csv', index=False)


