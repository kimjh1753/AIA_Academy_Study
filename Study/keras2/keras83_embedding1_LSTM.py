from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '안기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '규현이가 잘생기긴 했어요',
       ]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,1,0,0,0,0,0,1,1])
print(labels.shape) # (12,)

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x) # [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # pre: 앞쪽을 0으로 지정 post: 뒤쪽을 0으로 지정 maxlen: 정수, 모든 시퀀스의 최대 길이(길이 조절 해줌).
print(pad_x) 
print(pad_x.shape) # (13, 5)

print(np.unique(pad_x))
print(len(np.unique(pad_x))) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()
# model.add(Embedding(input_dim=28, output_dim=11, input_length=5)) 
# input_dim = 단어 사전의 개수 output_dim = 임의로 넣은 수(그 다음 레이어로 넘어가는 노드 수) input_length = 데이터의 길이(우리가 들어가야 할 최종 컬럼의 개수)         
model.add(Embedding(28, 11)) # Embedding layer에서 input_dim의 수는 크거나 같아야 한다.
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)
