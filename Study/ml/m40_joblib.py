from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# 1. Data
# x, y = load_boston(return_X_y=True)
datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle=True, random_state=66
)

# 2. Model
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=8)

# 3. Train
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'logloss', 'mae'], # 회귀 분석 문제는 rmse
          eval_set=[(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds=10)

aaa = model.score(x_test, y_test)
print("aaa : ", aaa)

y_pred = model.predict(x_test)
# r2 = r2_score(y_pred, y_test)
r2 = r2_score(y_test, y_pred)

print("r2 : ", r2)

print("=============================")
results = model.evals_result()
# print(results)

# aaa :  0.93302293398985
# r2 :  0.93302293398985

# 저장
import pickle
# pickle.dump(model, open('../data/xgb_save/m39.pickle.data', 'wb'))
# print('저장완료')
import joblib
# joblib.dump(model, "../data/xgb_save/m39.joblib.dat")

print("================= joblib 불러오기 ==================")
# 불러오기
# model2 = pickle.load(open('../data/xgb_save/m39.pickle.data', 'rb'))
model2 = joblib.load('../data/xgb_save/m39.joblib.dat')
print('불러왔다')
r22 = model2.score(x_test, y_test)
print('r22 : ', r22)

# ================= pickle 불러오기 ==================
# 불러왔다
# r22 :  0.93302293398985

# ================= joblib 불러오기 ==================
# 불러왔다
# r22 :  0.93302293398985