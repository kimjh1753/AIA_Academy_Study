import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.decomposition import PCA
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

# 1. Data
train = pd.read_csv('../study/DACON/data-1/train.csv')
test = pd.read_csv('../study/DACON/data-1/test.csv')
submission = pd.read_csv('../study/DACON/data-1/submission.csv')
print(train.shape, test.shape) # (2048, 787) (20480, 786)

# idx = 318
# img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
# digit = train.loc[idx, 'digit']
# letter = train.loc[idx, 'letter']

# plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
# plt.imshow(img)
# plt.show()

# 문자 데이터를 one-hot encoding하고
# 이미지 픽셀 데이터를 784개의 위치 feature라고 생각하고 concat
x_train = pd.concat(
    (pd.get_dummies(train.letter), train[[str(i) for i in range(784)]]), 
    axis=1)
y_train = train['digit']

# pca = PCA()
# x = pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum :",cumsum)

# d = np.argmax(cumsum >=0.97)+1
# print("cumsum >=0.95 :", cumsum>=0.97)
# print("d :", d) # 147

# Train set을 8:2로 분리
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01],
     "max_depth":[4,5,6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01]},
    {"max_depth": [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate" : [0.1, 0.001, 0.5],
     "max_depth": [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1],
     "colsample_bylevel" : [0.6, 0.7, 0.9]}
]
n_jobs = -1

# 2. Model
model = GridSearchCV(XGBClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                                         device='gpu', importance_type='split', learning_rate=0.1,
                                         max_depth=-1, min_child_samples=20, min_child_weight=0.001,
                                         min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=31,
                                         objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
                                         silent=True, subsample=1.0, subsample_for_bin=200000,
                                         subsample_freq=0, use_label_encoder=False), parameters, cv=kfold)

# 3. Train
model.fit(x_train, y_train, verbose=1, eval_metric=['merror', 'mlogloss', 'cox-nloglik'], 
          eval_set=[(x_train, y_train)],
          early_stopping_rounds=10)

# 예측 정확도 출력
print((model.predict(x_val) == y_val.values).sum() / len(y_val))

# Test 데이터에 대해 예측을 진행
x_test = pd.concat(
    (pd.get_dummies(test.letter), test[[str(i) for i in range(784)]]), 
axis=1)

# Submission 컬럼에 이를 기록
submission.digit = model.predict(x_test)

submission.to_csv('../study/DACON/data-1/xgb.csv', index=False)

# acc :  0.526829268292683



