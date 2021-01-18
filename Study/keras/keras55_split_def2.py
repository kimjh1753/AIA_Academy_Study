import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30]])

dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape) # dataset.shape :  (10, 3)

def split_xy3(dataset, x_col, x_row, y_columns, y_low):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_row = i + x_col 
        y_row = x_row + y_columns - 1

        if y_row > len(dataset):
            break
        tmp_x = dataset[i:x_row, : -1]
        tmp_y = dataset[x_row-1:y_row, -2:]
        x.append(tmp_x)
        y.append(tmp_y) 
    return np.array(x), np.array(y)
x, y = split_xy3(dataset, 3, 2, 1, 2)    
print(x, "\n", y)
print("x.shape : ", x.shape) # x.shape :  (8, 3, 2)
print("y.shape : ", y.shape) # y.shape :  (8, 1, 2)
print(type(x))
print(type(y))