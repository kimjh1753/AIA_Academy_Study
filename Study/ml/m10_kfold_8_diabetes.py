# 실습
# 6개의 모델을 완성하라!
# for문 쓸 수 있는 사람은 써봐라!! 후훗

import numpy as np
import warnings
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

print(x.shape, y.shape)      # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델 구성
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

# scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print('scores : ', scores)

'''
# model = LinearRegression()
# scores :  [0.55014272 0.49743662 0.38170042 0.4829625  0.50247162]

# model = KNeighborsRegressor()
# scores :  [0.32007098 0.31677384 0.37912411 0.42075977 0.54577744]

# model = DecisionTreeRegressor()
# scores :  [-0.38317325  0.09369005  0.26220018 -0.27015916 -0.29268422]

# model = RandomForestRegressor()
# scores :  [0.48137742 0.17769678 0.43730637 0.46709716 0.43486591]
'''

model = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
for i in range(4):
    scores = cross_val_score(model[i], x_train, y_train, cv=kfold)
    print('scores : ', scores)

# scores :  [0.50721149 0.57801005 0.40578743 0.40512248 0.51264439]
# scores :  [0.29598714 0.31465259 0.2385188  0.35773633 0.40829159]
# scores :  [-0.04369792 -0.16127378 -0.03435711 -0.24051868 -0.09695677]
# scores :  [0.45931018 0.47266655 0.3620639  0.49670562 0.23757285]

