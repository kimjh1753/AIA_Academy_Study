# 실습
# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 R2값과 피처임포턴스 구할것

# 2. 위 쓰레드 값으로 SelectFromModel을 구해서
# 최적의 피처 갯수를 구할 것

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치를 적용하여
# 최적의 R2 구할것

# 1번 값과 3번 값 비교

# 모델 : RandomForestRegressor

from inspect import Parameter
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

x, y = load_boston(return_X_y=True)

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

'''
thresholds = np.sort(model.best_estimator_.feature_importances_)
print(thresholds)
'''

# 최적의 매개변수 : XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.9,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=110, n_jobs=8, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# R2 :  0.9355635787909126
# [0.00210575 0.00531049 0.01011641 0.01048761 0.01293114 0.01457128
#  0.01917673 0.02380751 0.03694393 0.03844975 0.07241756 0.26056528
#  0.49311656]

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
# (404, 8)
# Thresh=0.015, n=8, R2: 93.52%


# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치를 적용하여
# 최적의 R2 구할것

thresholds = np.sort(model.best_estimator_.feature_importances_)
# print(thresholds) # 자르기 전

thresholds = thresholds[5:]
print(thresholds) # 자르기 후

# 최적의 매개변수 : XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=4,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=110, n_jobs=8, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# R2 :  0.9330675174335901
# [0.00209052 0.00401053 0.01096829 0.01161981 0.01329063 0.01581143
#  0.02356302 0.02380595 0.04629215 0.05074222 0.19108726 0.29350215
#  0.31321603]
# [0.01581143 0.02356302 0.02380595 0.04629215 0.05074222 0.19108726
#  0.29350215 0.31321603]

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

# (404, 8)
# Thresh=0.015, n=8, R2: 93.52%


# 최종 비교
# 1. 자르기 전
# (404, 8)
# Thresh=0.015, n=8, R2: 93.52%

# 2. 자르기 후
# (404, 8)
# Thresh=0.015, n=8, R2: 93.52%
