# 실습!!
# 전체 중요도에서 25% 미만인 컬럼들을 제거
# RandomForest로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
print(df.info()) 
# 필요한 열 번호 확인(전체 중요도에서 25% 이상인 컬럼) : 0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 

x = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 ]]
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
# [0.0214453  0.01618954 0.06204439 0.0569959  0.0084534  0.01816682
#  0.05579934 0.12471    0.00459907 0.00449379 0.01662765 0.00354237
#  0.00904798 0.03722194 0.00452086 0.00419953 0.00544213 0.00439084
#  0.0028785  0.00530924 0.09900462 0.02025132 0.11021475 0.12380309
#  0.01758359 0.01796517 0.03821369 0.08857513 0.0117195  0.00659056]
# acc :  0.9736842105263158
# ['symmetry error', 'smoothness error', 'compactness error', 'mean fractal dimension', 'mean symmetry', 'fractal dimension error', 'perimeter error', 'texture error']

# 데이터 자른 후
# [0.03030702 0.02189297 0.06154507 0.0329388  0.00810711 0.01131685
#  0.04386188 0.08607135 0.00266378 0.00372546 0.00810351 0.00385956
#  0.0116979  0.02599255 0.00554898 0.00532758 0.00598435 0.00391689
#  0.00299498 0.00539561 0.12561842 0.02097017 0.15897113 0.14146648
#  0.02164945 0.01933611 0.02342119 0.08750538 0.01085228 0.00895722]
# acc :  0.9649122807017544
# ['mean symmetry', 'symmetry error', 'mean fractal dimension', 'texture error', 'concave points error', 'compactness error', 'fractal dimension error', 'smoothness error']
