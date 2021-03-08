# 당뇨병 만들어 봐!!
# 0.5 이상!!!
# ㅇㅋ?

from inspect import Parameter
import numpy as np
from sklearn.datasets import load_boston, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

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

model = GridSearchCV(XGBRegressor(n_jobs=8), parameters, cv=kfold)

model.fit(x_train, y_train)

# 4. 평가, 예측
print("최적의 매개변수 :", model.best_estimator_)

score = model.score(x_test, y_test)
print("R2 : ", score) 

# thresholds = np.sort(model.best_estimator_.feature_importances_)
# print(thresholds)

# 최적의 매개변수 : XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
#              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=4,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=90, n_jobs=8, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# R2 :  0.3743166719670393
# [0.04509728 0.05277098 0.05327773 0.06255146 0.06481998 0.06695199
#  0.09276926 0.12383201 0.2188626  0.21906674]
'''
for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], 
          score*100))
'''
# (353, 5)
# Thresh=0.063, n=5, R2: 32.60%

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치를 적용하여
# 최적의 R2 구할것      

thresholds = np.sort(model.best_estimator_.feature_importances_)
print(thresholds) # 자르기 전

thresholds = thresholds[5:]
print(thresholds) # 자르기 후

# [0.03815481 0.04017618 0.05138198 0.05472512 0.06417811 0.0655996
#  0.06717855 0.11854213 0.17704022 0.32302332]
# [0.0655996  0.06717855 0.11854213 0.17704022 0.32302332]

for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], 
          score*100))

# (353, 5)
# Thresh=0.070, n=5, R2: 32.60%


# 최종 비교
# 1. 자르기 전
# (353, 5)
# Thresh=0.063, n=5, R2: 32.60%

# 2. 자르기 후
# (353, 5)
# Thresh=0.070, n=5, R2: 32.60%