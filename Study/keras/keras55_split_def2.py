import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                   [11,12,13,14,15,16,17,18,19,20],
                   [21,22,23,24,25,26,27,28,29,30],
                   [31,32,33,34,35,36,37,38,39,40]])

dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape : ", dataset.shape) # dataset.shape :  (10, 3)

def split_xy3(dataset, x_row, x_col, y_row, y_col):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_start_number = i                      # x 시작 지점
        x_end_number = i + x_row                # x 끝 지점 (row 길이를 잡아준다.)
        y_start_number = x_end_number           # y 시작 지점
        y_end_number = y_start_number + y_row   # y 끝 지점 (row 길이를 잡아준다.)

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[x_start_number : x_end_number, : x_col] # col 길이를 잡아준다.
        tmp_y = dataset[y_start_number : y_end_number, x_col : x_col + y_col] # # col 길이를 잡아준다.
        x.append(tmp_x)
        y.append(tmp_y) 
    return np.array(x), np.array(y)

x, y = split_xy3(dataset, 3, 2, 2, 2)    

print(x, "\n", y)
print("x.shape : ", x.shape) # x.shape :  (6, 3, 2)
print("y.shape : ", y.shape) # y.shape :  (6, 2, 2)

