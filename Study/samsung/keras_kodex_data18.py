import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/KODEX 코스닥150 선물인버스.csv', encoding = 'CP949', index_col=0, header=0, thousands=',',)

# 결측값이 들어있는 열 전체 제거 
df.drop(['전일비', 'Unnamed: 6'], axis=1, inplace=True)
print(df.info()) 

# 일자를 기준으로 오름차순
df = df.sort_values(by='일자', ascending=True)
print(df)

# 필요없는 열 전체 제거
df = df.drop(['금액(백만)','신용비','개인','기관','외인(수량)',
                  '외국계','프로그램','외인비'], axis=1)
print(df.info())
print(df.columns) # Index(['시가', '고가', '저가', '종가', '등락률', '거래량'], dtype='object')      

# 열 재배치
df = pd.DataFrame(df, columns=['고가', '저가', '종가', '등락률', '거래량', '시가'], dtype='float32')
print(df.info())
print(df.shape) # (1088, 6)
print(df)

data = df.to_numpy()
print(data)
print(type(data)) # <class 'numpy.ndarray'>
print(data.shape) # (1088, 6)

# np.save('./samsung/KODEX0118.npy', arr=data)

# size : 며칠 씩 자를 것인지
# col : 열의 개수

def split_x(seq, col, size) :
    dataset = []
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    # print(type(dataset))
    return np.array(dataset)

size = 6
col = 6
dataset = split_x(data, col, size)
print(dataset.shape) # (1083, 6, 6)

x2 = dataset[:-1,:,:]        
print(x2)
print(x2.shape)  # (1082, 6, 6) 

x2_pred = dataset[-1:,:,:] 
print(x2_pred)
print(x2_pred.shape) # (1, 6, 6)

np.save('./samsung/KODEX_data18.npy', arr=[x2, x2_pred])