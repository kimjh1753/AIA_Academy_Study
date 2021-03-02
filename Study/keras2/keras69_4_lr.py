x_train = 0.5
y_train = 0.8

######################### 얘네들 바꿔가면서 해봐 #############################
weights = 0.5
lr = 0.1
epoch = 500
######################### 얘네들 바꿔가면서 해봐 #############################

for iteration in range(epoch):
    y_predict = x_train * weights
    error = (y_predict - y_train) ** 2

    print("Error : " + str(error) + "\ty_predict : " + str(y_predict))

    up_y_predict = x_train * (weights + lr)
    up_error = (y_train - up_y_predict) ** 2

    down_y_predict = x_train * (weights - lr)
    down_error = (y_train - down_y_predict) ** 2

    if(down_error <= up_error):
        weights = weights - lr
    if(down_error > up_error):
        weights = weights + lr

# weights = 0.5
# lr = 0.1
# epoch = 1000
# Error : 1.232595164407831e-32   y_predict : 0.8000000000000002

               