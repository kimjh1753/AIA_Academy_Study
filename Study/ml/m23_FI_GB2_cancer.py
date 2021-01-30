# 실습!!
# 전체 중요도에서 상대적으로 점수가 낮은 컬럼들을 제거
# DesisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
print(df.info()) # 필요한 열 번호 확인(전체 중요도에서 상위인 컬럼) : 0, 1, 5, 7, 10, 11, 13, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28

x = df.iloc[:, [0, 1, 5, 7, 10, 11, 13, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28]]
print(x)
print(x.columns) # Index(['worst radius', 'worst texture', 'worst perimeter', 'worst smoothness','worst concave points'],dtype='object')
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 2. Model
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
model = GradientBoostingClassifier()

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

print(cut_columns(model.feature_importances_, dataset.feature_names, 10))

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
# [2.55510735e-04 5.05647457e-03 1.85675712e-03 1.06133894e-03
#  1.84196435e-05 2.61015980e-03 4.13617600e-04 4.05875400e-01
#  1.70840658e-03 1.77202141e-03 1.54414902e-03 3.20514223e-03
#  4.40552248e-04 9.31252152e-03 2.52099909e-04 1.71519512e-03
#  4.84495388e-04 9.87559185e-04 7.33082752e-04 2.18880239e-03
#  6.96119177e-02 6.56733598e-02 2.88671447e-01 4.29075556e-02
#  1.73902986e-02 5.32278621e-04 1.92772403e-02 5.21195388e-02
#  2.32081195e-03 3.84564961e-06]
# acc :  0.9736842105263158
# ['symmetry error', 'smoothness error', 'compactness error', 'mean fractal dimension', 'mean symmetry', 'fractal dimension error', 'perimeter error', 'texture error']

# 데이터 자른 후
# [6.85748503e-05 1.36440599e-02 6.87695438e-04 1.01399707e-03
#  1.84196435e-05 3.64743477e-03 6.01963743e-04 4.06381233e-01
#  1.49065658e-04 4.35438744e-03 2.04487109e-03 3.08982972e-03
#  1.29982621e-03 7.36837627e-03 1.33016589e-03 7.18651887e-04
#  4.26256885e-04 9.32007685e-04 1.20799310e-03 2.35053004e-03
#  7.46920613e-02 5.68085028e-02 2.89407031e-01 3.65755044e-02
#  1.32985324e-02 9.68626317e-04 1.96442619e-02 5.65152570e-02
#  7.52334434e-04 2.54815600e-06]
# acc :  0.9824561403508771
# ['worst fractal dimension', 'mean smoothness', 'mean radius', 'mean symmetry', 'concavity error', 'mean concavity', 'mean perimeter', 'compactness error', 'worst symmetry', 'concave points error']
