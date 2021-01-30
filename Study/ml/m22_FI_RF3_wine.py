# 실습!!
# 전체 중요도에서 25% 미만인 컬럼들을 제거
# RandomForest로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. Data
dataset = load_wine()

x = dataset['data']
y = dataset['target']

df = pd.DataFrame(x, columns=dataset['feature_names'])
# print(df)
print(df.columns)
print(df.info()) 
# 필요한 열 번호 확인(전체 중요도에서 25% 이상인 컬럼) : 0, 1, 5, 6, 7, 8, 9, 10, 11, 12 

x = df.iloc[:, [0, 1, 5, 6, 7, 8, 9, 10, 11, 12]]
print(x)
print(x.columns) # Index(['worst radius', 'worst texture', 'worst perimeter', 'worst smoothness','worst concave points'],dtype='object')
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 2. Model
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

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

print(cut_columns(model.feature_importances_, dataset.feature_names, 4))

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
# [0.13327507 0.03438444 0.01568483 0.02648967 0.02738152 0.0524495
#  0.18662907 0.01048864 0.02826106 0.13487081 0.07128013 0.09699144
#  0.18181382]
# acc :  0.9722222222222222
# ['nonflavanoid_phenols', 'ash', 'alcalinity_of_ash', 'magnesium']

# 데이터 자른 후
# [0.12823629 0.03324893 0.02049006 0.01839835 0.02405165 0.04735176
#  0.17291937 0.01106861 0.01963496 0.1271807  0.08982604 0.1190111
#  0.18858218]
# acc :  0.9444444444444444
# ['nonflavanoid_phenols', 'alcalinity_of_ash', 'proanthocyanins', 'ash'
