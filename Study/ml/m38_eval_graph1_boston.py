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
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'logloss', 'mae'], 
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

import matplotlib.pyplot as plt

epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.ylabel('XGBoost Log Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.ylabel('XGBoost Rmse')

plt.show()



