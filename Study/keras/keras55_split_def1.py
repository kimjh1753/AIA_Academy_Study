import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

# def split_xy2(dataset, time_steps, y_columns):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_columns
#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i : x_end_number]
#         tmp_y = dataset[x_end_number : y_end_number]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# time_steps = 4
# y_columns = 2
# x2, y2 = split_xy2(dataset, time_steps, y_columns)
# print(x2, "\n", y2)
# print("x2.shape : ", x2.shape)
# print("y2.shape : ", y2.shape)

def split_xy2(dataset, x_len, y_len):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_len
        y_end_number = x_end_number + y_len
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x_len = 4
y_len = 2
x2, y2 = split_xy2(dataset, x_len, y_len)
print(x2, "\n", y2)
print("x2.shape : ", x2.shape) # x2.shape :  (5, 4)
print("y2.shape : ", y2.shape) # y2.shape :  (5, 2)


# dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
#                    [11,12,13,14,15,16,17,18,19,20],
#                    [21,22,23,24,25,26,27,28,29,30]])
# print("dataset.shape : ", dataset.shape) # dataset.shape :  (3, 10)

# dataset = np.transpose(dataset)
# print(dataset)
# print("dataset.shape : ", dataset.shape) # dataset.shape :  (10, 3)

# def split_xy3(dataset, time_steps, y_columns):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_columns - 1

#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i:x_end_number, : -1]
#         tmp_y = dataset[x_end_number-1:y_end_number, -1]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)
# x, y = split_xy3(dataset, 3, 1)    
# print(x, "\n", y)
# print("x.shape : ", x.shape) # x.shape :  (8, 3, 2)
# print("y.shape : ", y.shape) # y.shape :  (8, 1)

# def split_xy3(dataset, time_steps, y_columns):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_columns - 1

#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i:x_end_number, : -1]
#         tmp_y = dataset[x_end_number-1:y_end_number, -1]
#         x.append(tmp_x)
#         y.append(tmp_y) 
#     return np.array(x), np.array(y)
# x, y = split_xy3(dataset, 3, 2)    
# print(x, "\n", y)
# print("x.shape : ", x.shape) # x.shape :  (7, 3, 2)
# print("y.shape : ", y.shape) # y.shape :  (7, 2)
# print(type(x))
# print(type(y))



