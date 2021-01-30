# 실습!!
# 피처임포턴스가 0인 컬럼들을 제거 하여 데이터셋을 재 구성후
# DesisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. Data
dataset = load_diabetes()

x = dataset['data']
y = dataset['target']

df = pd.DataFrame(x, columns=dataset['feature_names'])
# print(df)
print(df.columns)
print(df.info()) # 필요한 열 번호 확인(피처임포턴스가 0이 아닌 컬럼) : 0, 2, 4, 5, 8

x = df.iloc[:, [0, 2, 4, 5, 8]]
print(x)
print(x.columns) # Index(['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='object')
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 2. Model
model = DecisionTreeRegressor(max_depth=4)

# 3. Train
model.fit(x_train, y_train)

# 4. Evaluate, Predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_feautres = dataset.data.shape[1]
    plt.barh(np.arange(n_feautres), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_feautres), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_feautres)

plot_feature_importances_dataset(model)
plt.show()    

# 데이터 자르기 전
# [2.96958480e-02 0.00000000e+00 3.19268150e-01 1.28086228e-03
#  2.89697575e-02 4.99774661e-02 0.00000000e+00 2.16063823e-04
#  5.70591853e-01 0.00000000e+00]
# acc :  0.31490122539834386

# 데이터 자른 후
# [2.96958480e-02 0.00000000e+00 3.20549012e-01 0.00000000e+00
#  1.83192445e-02 6.06279792e-02 2.16063823e-04 0.00000000e+00
#  5.70591853e-01 0.00000000e+00]
# acc :  0.31490122539834386
