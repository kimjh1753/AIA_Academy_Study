import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/삼성전자.csv', encoding = 'CP949', index_col=0, header=0)
print(df)

df = df.fillna(method='pad')    # 각각의 결측치 바로 앞에 있는 value를 채워넣는 방식
# str.replace(',','').astype('int64') : str.replace(',','') -> ,를 지워준다   .astype('int64') -> int64형으로 바꿔준다.
df['시가'] =df['시가'].str.replace(',','').astype('int64')    
df['고가'] =df['고가'].str.replace(',','').astype('int64')
df['저가'] =df['저가'].str.replace(',','').astype('int64')
df['종가'] =df['종가'].str.replace(',','').astype('int64')
df['거래량'] =df['거래량'].str.replace(',','').astype('int64')
df['금액(백만)'] =df['금액(백만)'].str.replace(',','').astype('int64')
df['개인'] =df['개인'].str.replace(',','').astype('int64')
df['기관'] =df['기관'].str.replace(',','').astype('int64')
df['외인(수량)'] =df['외인(수량)'].str.replace(',','').astype('int64')
df['외국계'] =df['외국계'].str.replace(',','').astype('int64')
df['프로그램'] =df['프로그램'].str.replace(',','').astype('int64')
print(df.info())

x = df.iloc[:662, [0,1,2,3,4,5,8,9,10,11,12]]   # (662, 11)
y = df.iloc[:662 ,[3]]   # (662, 1)

print(df.shape) # (2400, 14)
print(x.shape) # (662, 11)
print(y.shape) # (662, 1)

aaa = x.to_numpy()
print(aaa)
print(type(aaa)) # <class 'numpy.ndarray'>
print(aaa.shape) # (662, 11)

bbb = y.to_numpy()
print(bbb)
print(type(bbb)) # <class 'numpy.ndarray'>
print(bbb.shape) # (662, 1)

np.save("../study/samsung/삼성전자_x.npy", arr=aaa)
np.save("../study/samsung/삼성전자_y.npy", arr=bbb)

# print(df.info())
# [2400 rows x 14 columns]
# <class 'pandas.core.frame.DataFrame'>
# Index: 2400 entries, 2021-01-13 to 2011-04-18
# Data columns (total 14 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   시가      2400 non-null   int64
#  1   고가      2400 non-null   int64
#  2   저가      2400 non-null   int64
#  3   종가      2400 non-null   int64
#  4   등락률     2400 non-null   float64
#  5   거래량     2400 non-null   int64
#  6   금액(백만)  2400 non-null   int64
#  7   신용비     2400 non-null   float64
#  8   개인      2400 non-null   int64
#  9   기관      2400 non-null   int64
#  10  외인(수량)  2400 non-null   int64
#  11  외국계     2400 non-null   int64
#  12  프로그램    2400 non-null   int64
#  13  외인비     2400 non-null   float64
# dtypes: float64(3), int64(11)
# memory usage: 281.2+ KB
# None
