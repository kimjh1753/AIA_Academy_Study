import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import warnings
from tensorflow.keras.backend import mean, maximum
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, Reshape
warnings.filterwarnings("ignore")

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
        return temp.iloc[-48:, :] # 7일 전부 다

df_train = preprocess_data(train)
print(df_train.shape) # (52464, 9)

# Index(['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T', 'Target1', 'Target2'], dtype='object')
dataset = df_train.to_numpy()

# 함수 : 시계열 데이터로 자르기
def split_xy(dataset, time_steps, y_row) :
    x, y1 = list(), list()
    for i in range(len(dataset)) :
        x_end = i + time_steps
        y_end = x_end
        if x_end > len(dataset) :
            break
        tmp_x = dataset[i:x_end, :-2]                # ['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']
        tmp_y1 = dataset[x_end-y_row : y_end, -2:]    # ['Target1', 'Target2']
        x.append(tmp_x)
        y1.append(tmp_y1)
    return np.array(x), np.array(y1)

x, y = split_xy(dataset, 48 , 48)
print(x.shape)     # (52417, 48, 7)  # day0 ~ day7, 7일씩 자름
# print(x[0:3])

print(y.shape)     # (52417, 48, 2)
# print(y[0:2])  

# test 데이터 불러오기 >> x_pred
df_test = []
for i in range(81):
    file_path = '../STUDY/DACON/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

df_test = pd.concat(df_test)
print(df_test.shape) # (3888, 7) -> 3888 = 81 * 48
x_pred = df_test.to_numpy()

x_pred = x_pred.reshape(int(x_pred.shape[0]/48), 48, x_pred.shape[1])
print(x_pred.shape) # (567, 48, 7)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])
print(x_pred.shape) # (567, 336)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape)    # (33546, 48, 7)
print(x_test.shape)     # (10484, 48, 7)
print(x_val.shape)      # (8387, 48, 7)

print(y_train.shape)    # (33546, 48, 2)
print(y_test.shape)     # (10484, 48, 2)
print(y_val.shape)      # (8387, 48, 2)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], 48*7)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(33546, 48, 7)
x_test = x_test.reshape(10484, 48, 7)
x_val = x_val.reshape(8387, 48, 7)
x_pred = x_pred.reshape(81, 48, 7)

y_train = y_train.reshape(33546, 48, 2)
y_test = y_test.reshape(10484, 48, 2)
y_val = y_val.reshape(8387, 48, 2)

# tensorflow pinball loss, Quantile loss definition
from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y, pred):
    err = (y - pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sub = pd.read_csv('../study/dacon/data/sample_submission.csv')

# 2. 모델 구성
def Mymodel():
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
    model.add(Dense(48*2, activation='relu'))
    model.add(Reshape((48,2)))
    model.add(Dense(2))
    return model

loss_list = list()

# 3. 컴파일, 훈련
for q in q_lst :
    # print("f'\n>>>>>>>>>>>>>>>>>>>>>> modeling start ('q_{q}')  >>>>>>>>>>>>>>>>>>>>>>" %q) 
    
    #2. Modeling
    model = Mymodel()
    model.summary()
    
    #3. Compile, Train
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(q, y, pred)])
    # modelpath = '../data/modelcheckpoint/dacon_solar_01_{epoch:02d}-{val_loss:.4f}.hdf5'
    # 02d = 정수 두 번째 자릿수까지 표기, .4f = 소수점 네 번째 자릿수까지 표기
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='auto')
    # cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, verbose=1)
    hist = model.fit(x_train, y_train, epochs=2, batch_size=32, callbacks=[early_stopping, reduce_lr], validation_data=(x_val, y_val))
    
    # 4. 평가, 예측 
    result = model.evaluate(x_test, y_test, batch_size=32)
    print("loss : ", result[0])
    print("loss : ", result[1])
    # print("('q_{q}') loss : %.4f" % (q, loss))
    # loss_list.append(loss)  # loss 기록

    y_pred = model.predict(x_pred)
    # print(y_pred.shape) # (81, 48, 2)
    y_pred = pd.DataFrame(y_pred.reshape(81*48, 2))    
    y_pred2 = pd.concat([y_pred], axis=1)
    y_pred2[y_pred < 0] =0
    y_pred3 = y_pred2.to_numpy()

     # submission
    column_name = 'q_' + str(q)
    sub.loc[sub.id.str.contains("Day7"), column_name] = y_pred3[:, 0].round(2)  # Day7 (3888, 9)
    sub.loc[sub.id.str.contains("Day8"), column_name] = y_pred3[:, 1].round(2)  # Day8 (3888, 9)

sub.to_csv('../STUDY/DACON/data/submission_data_5_1.csv', index=False)


