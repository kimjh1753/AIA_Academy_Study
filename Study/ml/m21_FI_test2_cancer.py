# 실습!!
# 피처임포턴스가 0인 컬럼들을 제거 하여 데이터셋을 재 구성후
# DesisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. Data
dataset = load_breast_cancer()

x = dataset['data']
y = dataset['target']

df = pd.DataFrame(x, columns=dataset['feature_names'])
# print(df)
print(df.columns)
print(df.info()) # 필요한 열 번호 확인(피처임포턴스가 0이 아닌 컬럼) : 20, 21, 22, 24 , 27

x = df.iloc[:, [20, 21, 22, 24, 27]]
print(x)
print(x.columns) # Index(['worst radius', 'worst texture', 'worst perimeter', 'worst smoothness','worst concave points'],dtype='object')
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 2. Model
model = DecisionTreeClassifier(max_depth=4)

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
# [0.         0.         0.         0.         0.         0.
#  0.00677572 0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.05612587 0.78000877 0.01008994
#  0.00995429 0.         0.         0.13026968 0.00677572 0.        ]
# acc :  0.9385964912280702

# 데이터 자른 후
# [569 rows x 5 columns]
# [0.         0.00677572 0.00677572 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.05612587 0.78000877 0.01008994
#  0.02293065 0.         0.         0.11729332 0.         0.        ]
# acc :  0.9385964912280702
