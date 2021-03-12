from tensorflow.keras.datasets import reuters, imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000
)

# print(x_train[0], type(x_train[0]))
# print(y_train[0])                       # 1
# print(len(x_train[0]), len(x_train[1])) # 218 189
print("================================")
print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)

# [실습 / 과제] Embedding으로 모델 만들 것!

# 1. 데이터
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')

print(x_train.shape, x_test.shape)  # (25000, 100) (25000, 100)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)  # (25000, 2) (25000, 2)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, LSTM, Conv1D

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64, input_length=100))
model.add(LSTM(32))
# model.add(Conv1D(32, 3))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.2)

result = model.evaluate(x_test, y_test)

print("loss : ", result[0]) 
print("acc : ", result[1])  

# loss :  1.5097426176071167
# acc :  0.8092399835586548