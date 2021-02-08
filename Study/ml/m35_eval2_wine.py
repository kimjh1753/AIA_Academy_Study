from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# 1. Data
# x, y = load_boston(return_X_y=True)
datasets = load_wine()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle=True, random_state=66
)

# 2. Model
model = XGBClassifier(n_estimators=10, learning_rate=0.01, n_jobs=8)

# 3. Train
model.fit(x_train, y_train, verbose=1, eval_metric='merror',  # 클래스 분류 문제는 error, 다중 클래스 분류 문제는 merror or mlogloss
          eval_set=[(x_train, y_train), (x_test, y_test)])

aaa = model.score(x_test, y_test)
print("aaa : ", aaa)

y_pred = model.predict(x_test)
# r2 = r2_score(y_pred, y_test)
r2 = r2_score(y_test, y_pred)

print("r2 : ", r2)

print("=============================")
results = model.evals_result()
print(results)

# aaa :  1.0
# r2 :  1.0
# =============================
# {'validation_0': OrderedDict([('merror', [0.007042, 0.007042, 0.007042, 0.007042, 0.007042, 0.007042, 0.007042, 0.007042, 0.007042, 0.007042])]), 
#  'validation_1': OrderedDict([('merror', [0.027778, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])}