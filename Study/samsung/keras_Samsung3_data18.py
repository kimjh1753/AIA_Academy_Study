import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/삼성전자.csv', encoding = 'CP949', index_col=0, header=0,  thousands=',', )
df1 = pd.read_csv('./samsung/삼성전자0115.csv', encoding = 'CP949', index_col=0, header=0, thousands=',',)

# 중복 데이터 제거
print(df.shape) # (2400, 14)
df = df.drop(['2021-01-13', '2018-05-03', '2018-05-02', '2018-04-30'])
print(df.shape) # (2396, 14)
print(df)

# 액면분할 전 가격 조정
df.iloc[661:, 0:4] = df.iloc[661:, 0:4]/50
df.iloc[661:, 5] = df.iloc[661:, 5]*50
print(df)

# 일자를 기준으로 오름차순
df = df.sort_values(by='일자' ,ascending=True)
print(df)

# 결측값이 들어있는 행 전체 제거
df1 = df1.rename(columns={'Unnamed: 6': '하락률'})
df1.drop(['전일비', '하락률'], axis=1, inplace=True)
print(df1)

# NULL 값 삭제
df1 = df1.dropna(axis=0)
print(df1.info())
# df1 = df1.astype('float32')
# print(df1.info())

# 일자를 기준으로 오름차순
df1 = df1.sort_values(by='일자' ,ascending=True)
print(df1)

# 필요없는 행 제거
df1 = df1.iloc[-3:,]
print(df1)
print(df)

data = pd.concat([df, df1])
print(data)
print(data.shape) # (2399, 14)
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

data = pd.DataFrame(data, columns=['고가', '저가', '등락률', '거래량', '종가', '시가'], dtype='float32')
print(data)
print(data.columns) # Index(['고가', '저가', '등락률', '거래량', '종가', '시가'], dtype='object')
print(data.shape) # (2399, 6)
print(data.info())

# KODEX와 행 맞추기
data = data.iloc[1311:,]
print(data.shape) # (1088, 6)
print(data)

final_data = data.to_numpy()
print(final_data)
print(type(final_data)) # <class 'numpy.ndarray'>
print(final_data.shape) # (1088, 6)

# np.save('./samsung/삼성전자0118.npy', arr=final_data)

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
print(dataset.shape) # (1083, 6, 6)

x1 = dataset[:-1,:,:]        
print(x1)
print(x1.shape)  # (1082, 6, 6) 

y1 = dataset[:-1,-1:,-2:]   
print(y1)
print(y1.shape)  # (1082, 1, 2)

x1_pred = dataset[-1:,:,:] 
print(x1_pred)
print(x1_pred.shape) # (1, 6, 6)

np.save('./samsung/samsung2_data18.npy', arr=[x1, y1, x1_pred])
