# 실습
# 6개의 모델을 완성하라!
# for문 쓸 수 있는 사람은 써봐라!! 후훗

import numpy as np
import warnings
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)

print(x.shape, y.shape)      # (506, 13) (506,)

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
# scores :  [0.69849307 0.55446771 0.72205133 0.68972357 0.73218051]

# model = KNeighborsRegressor()
# scores :  [0.52215881 0.37201905 0.47638623 0.37493395 0.37860922]

# model = DecisionTreeRegressor()
# scores :  [0.41817785 0.86856468 0.72652825 0.66275922 0.57587407]

# model = RandomForestRegressor()
# scores :  [0.78435647 0.86305448 0.89780261 0.84421241 0.89031591]
'''

model = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
for i in range(4):
    scores = cross_val_score(model[i], x_train, y_train, cv=kfold)
    print('scores : ', scores)

# scores :  [0.77958178 0.75639723 0.74039421 0.65628492 0.46979964]
# scores :  [0.41317482 0.34239862 0.58698611 0.46763828 0.45723993]
# scores :  [0.65751125 0.72570296 0.51748246 0.6452263  0.80750832]
# scores :  [0.91992743 0.82552256 0.90361183 0.70506899 0.81480873]
