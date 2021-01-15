import numpy as np
import pandas as pd

df = pd.read_csv('../study/samsung/삼성전자.csv', encoding = 'CP949', index_col=0, header=0,  thousands=',', )
df1 = pd.read_csv('../study/samsung/삼성전자2.csv', encoding = 'CP949', index_col=0, header=0, thousands=',',)

# 일자를 기준으로 오름차순
df = df.sort_values(by='일자' ,ascending=True)
print(df)

# 중복 데이터 제거
print(df.shape) # (2400, 14)
df = df.drop(['2021-01-13', '2018-05-03', '2018-05-02', '2018-04-30'])
print(df.shape) # (2396, 14)

# 결측값이 들어있는 행 전체 제거
df1 = df1.rename(columns={'Unnamed: 6': '하락률'})
df1.drop(['전일비', '하락률'], axis=1, inplace=True)
print(df1)

# 일자를 기준으로 오름차순
df1 = df1.sort_values(by='일자' ,ascending=True)
print(df1)

# NULL 값 삭제
df1 = df1.dropna(axis=0)
print(df1.info())
# df1 = df1.astype('float32')
# print(df1.info())

df.iloc[661:, 0:4] = df.iloc[661:, 0:4]/50
df.iloc[661:, 5] = df.iloc[661:, 5]/50

data = pd.concat([df, df1])
print(data)
print(data.shape) # (2398, 14)
print(data.info())

# # 시각화1  : 상관계수 히트맵
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.8, font='Malgun Gothic', rc={'axes.unicode_minus':False})
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
#     # heatmap : 사각형 형태로 만들겠다.
#     # 데이터 : df.corr()
#     # square=True : 사각형 형태로 표현
#     # annot=True : 글씨를 넣겠다.
#     # cbar=True : 옆에 있는 바를 넣겠다.
# plt.show()

data = data.drop(['금액(백만)','신용비','개인','기관','외인(수량)',
                  '외국계','프로그램','외인비'], axis=1)
print(data)
print(data.columns) # Index(['시가', '고가', '저가', '종가', '등락률'], dtype='object')
print(data.info())

data = pd.DataFrame(data, columns=['시가', '고가', '저가', '등락률', '거래량', '종가'], dtype='float32')
print(data)
print(data.columns) # Index(['시가', '고가', '저가', '등락률', '종가'], dtype='object')
print(data.shape) # (2398, 6)
print(data.info())

final_data = data.to_numpy()
# print(final_data)
# print(type(final_data)) # <class 'numpy.ndarray'>
# print(final_data.shape) # (2398, 6)

# np.save('./samsung/삼성전자0115.npy', arr=final_data)

# size : 며칠씩 자를 것인지
# col : 열의 개수

def split_x(seq, col,size) :
    dataset = []  
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    # print(type(dataset))
    return np.array(dataset)

size = 6
col = 6
dataset = split_x(final_data,col,size)
print(dataset.shape) # (2393, 6, 6)

x = dataset[1:,:,:]        
print(x)
print(x.shape)  # (2392, 6, 6) 

y = dataset[1:,-1:,-1:]   
print(y)
print(y.shape)  # (2392, 1, 1)

x_pred = dataset[-1:,:,:] 
print(x_pred)
print(x_pred.shape) # (1, 6, 6)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, shuffle=True, random_state=311
)
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

print(x_train.shape)    # (1530, 6, 6)
print(x_test.shape)     # (479, 6, 6)
print(x_val.shape)      # (383, 6, 6) 

y_train = y_train.reshape(y_train.shape[0] ,1)
y_test = y_test.reshape(y_test.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)

print(y_train.shape) # (1530, 1)
print(y_test.shape)  # (479, 1)
print(y_val.shape)   # (381, 1)

# 전처리(3차원->2차원)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])   
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], size*col)

print(x_train.shape)    # (1532, 36)
print(x_test.shape)     # (479, 36)
print(x_val.shape)      # (383, 36) 
print(x_pred.shape)     # (1, 36)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_test.shape[1])

# 전처리(2차원-> 3차원)
x_train = x_train.reshape(x_train.shape[0], size, col)   
x_test = x_test.reshape(x_test.shape[0], size, col)
x_val = x_val.reshape(x_val.shape[0], size, col)
x_pred = x_pred.reshape(x_pred.shape[0], size, col)

print(x_train.shape)        # (1530, 6, 6)
print(x_test.shape)         # (479, 6, 6)
print(x_val.shape)          # (383, 6, 6)
print(x_pred.shape)         # (1, 6, 6)

np.save('./samsung/samsung2_data15.npy', arr=[x_train, x_test, x_val, y_train, y_test, y_val, x_pred])



