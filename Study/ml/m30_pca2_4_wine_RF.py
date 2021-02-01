# 랜덤포레스트로 모델링 하시오.

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_wine()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (178, 13) (178,)

# pca = PCA(n_components=11)
# x2 = pca.fit_transform(x)
# print(x2)
# print(x2.shape)         # (442, 7)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR) 
# print(sum(pca_EVR))

# 7 : 0.9479436357350414
# 8 : 0.9913119559917797
# 9 : 0.9991439470098977
# 10 : 1.0

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) # cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
print("cumsum : ", cumsum)
# cumsum :  [0.99809123 0.99982715 0.99992211 0.99997232 0.99998469 0.99999315
#  0.99999596 0.99999748 0.99999861 0.99999933 0.99999971 0.99999992
#  1.        ]

d = np.argmax(cumsum > 0.95)+1
print("cumsum >= 0.95", cumsum >= 0.95) 
# cumsum >= 0.95 [ True  True  True  True  True  True  True  True  True  True  True  True True]
print("d : ", d) # d :  1

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

pca = PCA(n_components=d)
x = pca.fit_transform(x)
# print(x)
print(x.shape)         # (178, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12], 'n_estimators' : [100, 200]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_jobs' : [-1, 2, 4], 'n_estimators' : [100, 200]}
]

# 2. 모델 구성
# model = SVC()
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print("최적의 매개변수 :", model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률', r2_score(y_test, y_pred))

aaa = model.score(x_test, y_test)
print(aaa)

# 최적의 매개변수 : RandomForestClassifier(n_estimators=200, n_jobs=-1)
# 최종정답률 -0.18867924528301905
# 0.5555555555555556
