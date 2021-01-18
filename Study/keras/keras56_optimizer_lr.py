import numpy as np

# 1. Data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. Model
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. Compile, Train
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# Adam
# optimizer = Adam(lr=0.1)
# loss :  2.2714266023626806e-08 결과물 :  [[10.999332]]
# optimizer = Adam(lr=0.01)
# loss :  1.4530599488395135e-13 결과물 :  [[11.]]
# optimizer = Adam(lr=0.001)
# loss :  1.2362719774246216 결과물 :  [[12.975614]]

# Adaelta
# optimizer = Adadelta(lr=0.1)
# loss :  3.7793022784171626e-05 결과물 :  [[10.990527]]
# optimizer = Adadelta(lr=0.01)
# loss :  8.680133760208264e-05 결과물 :  [[10.988011]]
# optimizer = Adadelta(lr=0.001)
# loss :  8.061490058898926 결과물 :  [[5.901609]]

# Adamax
# optimizer = Adamax(lr=0.1)
# loss :  1.0978891573643068e-08 결과물 :  [[11.000138]]
# optimizer = Adamax(lr=0.01)
# loss :  1.5322853662985692e-12 결과물 :  [[11.]]
# optimizer = Adamax(lr=0.001)
# loss :  5.31832654360187e-07 결과물 :  [[10.999164]]

# Adagrad
# optimizer = Adagrad(lr=0.1)
# loss :  3.309366729808971e-05 결과물 :  [[11.006962]]
# optimizer = Adagrad(lr=0.01)
# loss :  2.7765897812059848e-06 결과물 :  [[10.997577]]
# optimizer = Adagrad(lr=0.001)
# loss :  7.974177424330264e-05 결과물 :  [[10.983921]]

# RMSprop
# optimizer = RMSprop(lr=0.1)
# loss :  250325216.0 결과물 :  [[32452.283]]
# optimizer = RMSprop(lr=0.01)
# loss :  41.14154052734375 결과물 :  [[20.941631]]
# optimizer = RMSprop(lr=0.001)
# loss :  0.004271487705409527 결과물 :  [[10.858815]]

# SGD
# optimizer = RMSprop(lr=0.1)
# loss :  61939568640.0 결과물 :  [[346422.97]]
# optimizer = RMSprop(lr=0.01)
# loss :  1.353057861328125 결과물 :  [[8.898738]]
# optimizer = RMSprop(lr=0.001)
# loss :  0.6308239102363586 결과물 :  [[12.279798]]

# Nadam
# optimizer = RMSprop(lr=0.1)
# loss :  152732.28125 결과물 :  [[762.09357]]
# optimizer = RMSprop(lr=0.01)
# loss :  0.719258725643158 결과물 :  [[12.2858925]]
optimizer = RMSprop(lr=0.01)
# loss :  78.22566223144531 결과물 :  [[26.41529]]

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

# 4. Evaluate, Predict
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
print("loss : ", loss, "결과물 : ", y_pred)
