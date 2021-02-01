# 실습!!
# 전체 중요도에서 상대적으로 점수가 낮은 컬럼들을 제거
# DesisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pandas as pd
import numpy as np

# 1. Data
dataset = load_wine()

x = dataset['data']
y = dataset['target']

df = pd.DataFrame(x, columns=dataset['feature_names'])
# print(df)
print(df.columns)
print(df.info()) # 필요한 열 번호 확인(전체 중요도에서 상위인 컬럼) : 1, 2, 6, 9, 10, 11, 12

x = df.iloc[:, [1, 2, 6, 9, 10, 11, 12]]
print(x)
print(x.columns) # Index(['worst radius', 'worst texture', 'worst perimeter', 'worst smoothness','worst concave points'],dtype='object')
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 2. Model
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs=-1)

# 3. Train
model.fit(x_train, y_train)

# 4. Evaluate, Predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

# 가장 중요도 낮은피쳐 number만큼 반환해주는 함수만들었어요 참고하세요!
def cut_columns(feature_importances,columns,number):
    temp = []
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result

print(cut_columns(model.feature_importances_, dataset.feature_names, 7))

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
# [6.06643138e-02 3.07521737e-02 1.35318277e-02 1.65494784e-03
#  7.96488236e-03 1.48270097e-07 2.15133308e-01 1.54924346e-03
#  5.57316630e-05 2.31899730e-01 1.92991802e-02 1.25667327e-01
#  2.91827186e-01]
# acc :  0.9444444444444444
# ['total_phenols', 'proanthocyanins', 'nonflavanoid_phenols', 'alcalinity_of_ash', 'magnesium', 'ash', 'hue']

# 데이터 자른 후
# [5.88679058e-02 3.74690305e-02 1.03975759e-02 1.57880966e-04
#  7.96221009e-03 1.60749695e-07 1.96849215e-01 2.53294403e-03
#  9.38833445e-04 2.21630287e-01 1.93252947e-02 1.50641570e-01
#  2.93227091e-01]
# acc :  0.9166666666666666
# ['total_phenols', 'alcalinity_of_ash', 'proanthocyanins', 'nonflavanoid_phenols', 'magnesium', 'ash', 'hue']

# xgb
# [0.06830431 0.04395564 0.00895185 0.         0.01537293 0.00633501
#  0.07699447 0.00459099 0.00464443 0.08973485 0.01806366 0.5588503
#  0.10420163]
# acc :  0.9444444444444444
# ['alcalinity_of_ash', 'nonflavanoid_phenols', 'proanthocyanins', 'total_phenols', 'ash', 'magnesium', 'hue']