# 실습!!
# 전체 중요도에서 상대적으로 점수가 낮은 컬럼들을 제거
# DesisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pandas as pd
import numpy as np
# import warnings
# warnings.filterwarnings('ignore')

# 1. Data
dataset = load_iris()

x = dataset['data']
y = dataset['target']

df = pd.DataFrame(x, columns=dataset['feature_names'])
# print(df)
print(df.columns)
print(df.info()) # 필요한 열 번호 확인(전체 중요도에서 상위인 컬럼) : 2, 3

x = df.iloc[:, [2, 3]]
print(x)
print(x.columns) # Index(['worst radius', 'worst texture', 'worst perimeter', 'worst smoothness','worst concave points'],dtype='object')
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 타임 걸어
import datetime
date_now = datetime.datetime.now()
print(date_now)  

# 2. Model
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs=8, n_estimators=100)

# 3. Train
model.fit(x_train, y_train, eval_metric='mlogloss', verbose=True,    # logloss
          eval_set=[(x_train, y_train), (x_test, y_test)]
) 

# 4. Evaluate, Predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

# 타임 걸어
# n_jobs = -1, 8, 4, 1 속도 비교
date_now = datetime.datetime.now()
print(date_now) 

# n_jobs = -1 (모델전 -> acc 다음) 2021-02-01 10:48:13.978195 -> 2021-02-01 10:48:44.225856
# n_jobs = 8  (모델전 -> acc 다음) 2021-02-01 10:50:06.778847 -> 2021-02-01 10:50:06.853178
# n_jobs = 4  (모델전 -> acc 다음) 2021-02-01 10:51:45.447574 -> 2021-02-01 10:51:45.530308
# n_jobs = 1  (모델전 -> acc 다음) 2021-02-01 10:52:12.778214 -> 2021-02-01 10:52:12.844750

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

print(cut_columns(model.feature_importances_, dataset.feature_names, 2))

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
# [0.00806173 0.01142893 0.72461193 0.2558974 ]
# acc :  0.9666666666666667
# ['sepal length (cm)', 'sepal width (cm)']

# 데이터 자른 후
# [0.00799375 0.01126205 0.6542637  0.3264805 ]
# acc :  0.9666666666666667
# ['sepal length (cm)', 'sepal width (cm)']

# xgb
# [0.02323038 0.01225644 0.8361378  0.12837538]
# acc :  0.9666666666666667
# ['sepal width (cm)', 'sepal length (cm)']