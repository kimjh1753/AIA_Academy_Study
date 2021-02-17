import numpy as np
import keras
import cv2, os

# class DataGenerator(keras.utils.Sequence) => '클래스의 생성자 입니다. 클래스를 생성할 때 인자들을 받아옵니다. 
# 클래스를 생성시 default값이 존재하지 않는이상 반드시 입력을 해주어야 하며, 클래스 생성과 동시에 작동하는 함수입니다.'
class DataGenerator(keras.utils.Sequence): 
    'Generates data for Keras' 
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization' # ' 초기화 '
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    # def __len__(self) => 길이를 호출하는 함수, return값으로 길이를 반환
    def __len__(self): 
        'Denotes the number of batches per epoch' #  epoch 당 배치 수를 나타냅니다 '
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index): # '실질적으로 배치를 반환하는 부분'
        'Generate one batch of data' 
        # Generate indexes of the batch 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs = IDs 의 리스트(목록) 찾기
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data = 데이터 생성
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self): # '한 epoch을 수행한 후에 fit_generator함수 안에서 호출되는 함수'
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.dim[0] * 4, self.dim[1] * 4, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = np.load(ID)

            # ex) C:\Project\celeba-dataset\processed\x_test 
            splited = ID.split('\\') # -> ["C:", "Project", "celeba-dataset", "processed", "x_test"]
            splited[-2] = 'y' + splited[-2][1:] 
            # x_train -> y_train : 경로에서 마지막 2번째 "processed"에서 y로 시작하는 폴더에서 첫번째 행을 제외한 나머지를 가져와서 데이터 생성
            splited[0] = splited[0] + os.sep # os.sep : pathname 분리기호 ('/' or '\')
            y_path = os.path.join(os.sep, *splited) # 경로를 병합하여 새 경로 생성

            # Store class
            y[i] = np.load(y_path)

        return X, y