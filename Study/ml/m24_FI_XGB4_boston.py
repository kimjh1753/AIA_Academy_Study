# 실습!!
# 전체 중요도에서 상대적으로 점수가 낮은 컬럼들을 제거
# DesisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import pandas as pd
import numpy as np

# 1. Data
dataset = load_boston()

x = dataset['data']
y = dataset['target']

df = pd.DataFrame(x, columns=dataset['feature_names'])
# print(df)
print(df.columns)
print(df.info()) # 필요한 열 번호 확인(전체 중요도에서 상위인 컬럼) : 4, 5, 6, 7, 10, 12

x = df.iloc[:, [4, 5, 6, 7, 10, 12]]
print(x)
print(x.columns) # Index(['worst radius', 'worst texture', 'worst perimeter', 'worst smoothness','worst concave points'],dtype='object')
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 타임 걸어

# 2. Model
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingRegressor()
model = XGBRegressor(n_jobs=-1) # 

# 3. Train
model.fit(x_train, y_train)

# 4. Evaluate, Predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

# 타임 걸어

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

print(cut_columns(model.feature_importances_, dataset.feature_names, 8))

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
# [2.98990563e-02 2.89676686e-04 2.36114857e-03 1.13227503e-03
#  3.09298122e-02 3.79663002e-01 8.61388214e-03 9.80112981e-02
#  6.79904846e-04 1.17455304e-02 3.47636593e-02 5.47186974e-03
#  3.96438885e-01]
# acc :  0.8954521943395123
# ['ZN', 'RAD', 'CHAS', 'INDUS', 'B', 'AGE', 'TAX', 'CRIM'] 4, 5, 6, 7, 10, 12

# 데이터 자른 후
# [0.02859324 0.0004277  0.00262071 0.00113228 0.03260204 0.37922774
#  0.00852187 0.0978224  0.00079086 0.0114876  0.03288188 0.00727925
#  0.39661245]
# acc :  0.8917795387259541
# ['ZN', 'RAD', 'CHAS', 'INDUS', 'B', 'AGE', 'TAX', 'CRIM']

# xgb
# [0.01311134 0.00178977 0.00865051 0.00337766 0.03526587 0.24189197
#  0.00975884 0.06960727 0.01454236 0.03254252 0.04658296 0.00757505
#  0.51530385]
# acc :  0.8902902185916939
# ['ZN', 'CHAS', 'B', 'INDUS', 'AGE', 'CRIM', 'RAD', 'TAX']