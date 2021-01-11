# cnn
'''
    Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu')

    첫번째 인자 : 컨볼루션 필터의 수 입니다.
    두번째 인자 : 컨볼루션 커널의 (행, 열) 입니다.
    padding : 경계 처리 방법을 정의합니다.
        -‘valid’ : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
        -‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.
    input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다.
        (행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
    activation : 활성화 함수 설정합니다.
        ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
        ‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
        ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
        ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1,      # strides = 2 -> 시작 지점부터 2칸씩 간 뒤에 자른다
                 padding='same', input_shape=(10, 10, 1)))
# model.add(MaxPooling2D(pool_size=1))                               
# model.add(MaxPooling2D(pool_size=2))                
model.add(MaxPooling2D(pool_size=(2,3)))                
model.add(Conv2D(9, (2,2), padding='valid'))
# model.add(Conv2D(9, (2,3)))
# model.add(Conv2D(8, 2))
model.add(Flatten())        # Faltten() -> cnn 4차원을 2차원으로 바꿔준다.
model.add(Dense(1))

model.summary()

# model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', input_shape=(10, 10, 1)))
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 10, 10, 10)        50                 (None, 10, 10, 10) => padding = 'same'
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 9, 9, 9)           369
# _________________________________________________________________
# flatten (Flatten)            (None, 729)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 730
# =================================================================
# Total params: 1,149
# Trainable params: 1,149
# Non-trainable params: 0

# Parameter = (input_dim x kernal_size + bias) x filter_output
# (1 x 2 x 2 + 1) x 10 = 50
# (10 x 2 x 2 + 1) x 9 = 369

# model.add(MaxPooling2D(pool_size=1))                               
# conv2d (Conv2D)              (None, 10, 10, 10)        50
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 10, 10, 10)        0        

# model.add(MaxPooling2D(pool_size=2))                
# conv2d (Conv2D)              (None, 10, 10, 10)        50
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 5, 5, 10)          0

# model.add(MaxPooling2D(pool_size=(2,3)))                
# conv2d (Conv2D)              (None, 10, 10, 10)        50
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 5, 3, 10)          0
