# 실습!!
# 전체 중요도에서 25% 미만인 컬럼들을 제거
# RandomForest로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 1. Data
dataset = load_boston()

x = dataset['data']
y = dataset['target']

df = pd.DataFrame(x, columns=dataset['feature_names'])
# print(df)
print(df.columns)
print(df.info()) 
# 필요한 열 번호 확인(전체 중요도에서 25% 이상인 컬럼) : 0, 4, 5, 6, 7, 9, 10, 11, 12

x = df.iloc[:, [0, 4, 5, 6, 7, 9, 10, 11, 12]]
print(x)
print(x.columns) # Index(['worst radius', 'worst texture', 'worst perimeter', 'worst smoothness','worst concave points'],dtype='object')
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 2. Model
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestRegressor()

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
# [0.03728387 0.00073951 0.00554173 0.00092239 0.02315869 0.36845713
#  0.01766227 0.07571442 0.00309674 0.01387721 0.01977072 0.01129245
#  0.42248288]
# acc :  0.8902547367482933
# ['ZN', 'CHAS', 'RAD', 'INDUS']

# 데이터 자른 후
# [0.04158689 0.00078634 0.00611266 0.0008741  0.0252228  0.40813705
#  0.01401957 0.06726812 0.00260644 0.01495059 0.01835022 0.00941426
#  0.39067097]
# acc :  0.8831174670217627
# ['ZN', 'CHAS', 'RAD', 'INDUS']
