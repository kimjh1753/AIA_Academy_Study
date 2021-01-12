import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)
print(x)
print(y)


# 전처리 알아서 해 / MinMaxScaler, train_test_split
print(np.max(x[0]))

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)
x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size = 0.8, random_state = 66, shuffle = True
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape)
print(y_train.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(30,))
aaa = Dense(1000, activation='relu')(input1)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
aaa = Dense(1000, activation='relu')(aaa)
outputs = (Dense(1, activation='sigmoid'))(aaa)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# 3. 컴파일, 훈련
                   # mean_squared_error
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto') 
modelpath = '../data/modelcheckpoint/k46_MC_6_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, epochs=2000, validation_data=(x_val, y_val), 
               callbacks=[early_stopping, cp], verbose=1)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss, accuracy : ", result[0], result[1])

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6)) # 단위 알아서 찾을 것!

plt.subplot(2, 1, 1)    # 2행 1열중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

# plt.title('Cost loss')    # 한글깨짐 오류 해결할 것 과제1.
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 2행 2열중 두번째
plt.plot(hist.history['accuracy'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue', label='val_accuracy')
plt.grid()

# plt.title('정확도')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# keras Dense cancer
# loss, acc :  1.0589289665222168 0.9122806787490845
# [[1.       ]
#  [0.0028897]
#  [1.       ]
#  [1.       ]]
# [1 0 1 1]

# keras MC_6_cancer
# loss, accuracy :  3.204373359680176 0.9649122953414917
# [[1.]
#  [0.]
#  [1.]
#  [1.]]
# [1 0 1 1]
