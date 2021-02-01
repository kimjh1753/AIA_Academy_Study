# m31로 만든 0.95 이상의 n_component=? 를 사용하여 
# xgb 모델을 만들것 (디폴트)

# mnist dnn보다 성능 좋게 만들어라!!!
# cnn과 비교!!!

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost.sklearn import XGBClassifier

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape) # (70000, 28, 28)
print(y.shape) # (70000,)

x = x.reshape(70000, 28*28)
print(x.shape) # (70000, 784)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) # cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
print("cumsum : ", cumsum) 

d = np.argmax(cumsum > 0.95)+1
print("cumsum >= 0.95", cumsum >= 0.95) 
print("d : ", d) # d :  154

pca = PCA(n_components=d)
x = pca.fit_transform(x)
# print(x)
print(x.shape) # (70000, 154)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
print(x_train.shape, x_test.shape) # (56000, 154) (14000, 154)
print(y_train.shape, y_test.shape) # (56000,) (14000,)

import warnings
warnings.filterwarnings('ignore')

kfold = KFold(n_splits=5, shuffle=True)

model = XGBClassifier(n_jobs=8, use_label_encoder=False)

# 3. Train
model.fit(x_train, y_train, eval_metric='logloss')

# 4. Evaluate, Predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc : ", acc)

# keras40_mnist2_cnn
# loss :  0.00260396976955235
# acc :  0.9854999780654907
# [[8.6690171e-08 2.8707976e-08 9.1137373e-09 9.6521189e-06 4.6547077e-09
#   9.9998856e-01 7.6187533e-08 5.5741470e-08 1.3864026e-06 2.0224462e-07]]
# ============
# [[7.0327958e-30 2.2413428e-23 6.9391834e-21 9.2217209e-22 5.1841172e-22
#   8.7506048e-26 2.4799229e-27 1.0000000e+00 8.0364114e-26 3.3208760e-17]]

# m33_pca_mnist1_xgb
# acc :  0.9656428571428571
