import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (506, 13) (506,)

from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = train_test_split(
    x, y, shuffle=True
)
 
print(x_train.shape, y_train.shape) # (404, 13) (102, 13)
print(x_test.shape, y_test.shape)   # (404,) (102,)

train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))

model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=1
)

model.fit(train_set, epochs=1, validation_split=0.2)
result = model.evaluate(test_set)

print(result)
