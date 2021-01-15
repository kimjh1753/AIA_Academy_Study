import numpy as np
import pandas as pd

df = pd.read_csv('../study/samsung/삼성전자.csv', encoding = 'CP949', index_col=0, header=0, thousands=',', )
df1 = pd.read_csv('../study/samsung/삼성전자2.csv', encoding = 'CP949', index_col=0, header=0, thousands=',',)

# str.replace(',','').astype('int64') : str.replace(',','') -> ,를 지워준다   .astype('int64') -> int64형으로 바꿔준다.
df = df.astype('float32')
print(df.info())

# 중복 데이터 제거
print(df.shape) # (2400, 14)
df = df.drop(['2021-01-13'])
print(df.shape) # (2399, 14)

df1 = df1.rename(columns={'Unnamed: 6': '하락률'})
df1.drop(['전일비', '하락률'], axis=1, inplace=True)
print(df1)
df1 = df1.dropna(axis=0)
# print(df1.info())
df1 = df1.astype('float32')
print(df1.info())

data = pd.concat([df1, df])
print(data)
print(data.shape) # (2401, 14)

# data = df.to_numpy()
# print(data)
# print(type(data)) # <class 'numpy.ndarray'>
# print(data.shape) # (2399, 14)

np.save("../study/samsung/삼성전자0114.npy", arr=data)

