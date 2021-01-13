import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 
#            'DESCR', 'feature_names', 'filename'])
print(dataset.values())
print(dataset.target_names)
# ['setosa' 'versicolor' 'virginica']

# x = dataset.data
x = dataset['data']
# y = dataset.target
y = dataset['target']
print(x)
print(y)
print(x.shape, y.shape) # (150, 4) (150,)
print(type(x), type(y)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# df = pd.DataFrame(x, columns=dataset.feature_names)
df = pd.DataFrame(x, columns=dataset['feature_names'])
print(df)
print(df.shape) # (150, 4)
print(df.columns) # Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                         # 'petal width (cm)'], dtype='object')
print(df.index) # RangeIndex(start=0, stop=150, step=1) -> 명시 안해주면 자동으로 인덱싱해준다.

print(df.head()) # df[:5]
print(df.tail()) # df[-5:]
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 4 columns):
#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   sepal length (cm)  150 non-null    float64
#  1   sepal width (cm)   150 non-null    float64
#  2   petal length (cm)  150 non-null    float64
#  3   petal width (cm)   150 non-null    float64
# dtypes: float64(4)
# memory usage: 4.8 KB
# None
print(df.describe())
#        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# count         150.000000        150.000000         150.000000        150.000000
# mean            5.843333          3.057333           3.758000          1.199333
# std             0.828066          0.435866           1.765298          0.762238
# min             4.300000          2.000000           1.000000          0.100000
# 25%             5.100000          2.800000           1.600000          0.300000
# 50%             5.800000          3.000000           4.350000          1.300000
# 75%             6.400000          3.300000           5.100000          1.800000
# max             7.900000          4.400000           6.900000          2.500000

df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(df.columns)
print(df.info())
print(df.describe())

# y칼럼을 추가해 보아요
print(df['sepal_length'])
df['Target'] = dataset.target
print(df.head())

print(df.shape) # (150, 5)
print(df.columns) # Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Target'], dtype='object')
print(df.index)
print(df.tail())

print(df.info())
print(df.isnull())
print(df.isnull().sum())
print(df.describe())
print(df['Target'].value_counts())

# 상관계수, 히트맵
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 도수 분포도
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.hist(x = 'sepal_length', data = df)
plt.title('sepal_length')

plt.subplot(2, 2, 2)
plt.hist(x = 'sepal_width', data = df)
plt.title('sepal_width')

plt.subplot(2, 2, 3)
plt.hist(x = 'petal_length', data = df)
plt.title('petal_length')

plt.subplot(2, 2, 4)
plt.hist(x = 'petal_width', data = df)
plt.title('petal_width')

plt.show()
