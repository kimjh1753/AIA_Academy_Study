x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

print(x, "\n", y)

import matplotlib.pyplot as plt
plt.plot(x, y)
# plt.show()

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})
print(df)
#     X   Y
# 0  -3  -2
# 1  31  32
# 2 -11 -10
# 3   4   5
# 4   0   1
# 5  22  23
# 6  -2  -1
# 7  -5  -4
# 8 -25 -24
# 9 -14 -13
print(df.shape) # (10, 2)

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']
print(x_train.shape, y_train.shape) # (10,) (10,)   (10,) -> 스칼라가 10개
print(type(x_train))                # <class 'pandas.core.series.Series'>

x_train = x_train.values.reshape(len(x_train), 1)   # (10,) -> (10, 1)
print(x_train.shape, y_train.shape)                 # (10, 1) (10,)   (10,) -> 스칼라가 10개

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print("score : ", score)    # score :  1.0

print("기울기(weight) : ", model.coef_)    # 기울기(weight) :  [2.]
print("절편(bias) : ",model.intercept_)    # 절편(bias) :  3.0