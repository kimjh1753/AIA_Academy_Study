import numpy as np

f = lambda x : x**2 - 4*x + 6

gradient = lambda x : 2*x -4

x0 = 5.0
epoch = 30
learning_rate = 10

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

for i in range(epoch):
    temp = x0 - learning_rate * gradient(x0)
    x0 = temp

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))

# x0 = 10.0
# epoch = 30
# learning_rate = 0.1
# 30      2.00990 2.00010

# x0 = 10.0
# epoch = 300
# learning_rate = 0.01    
# 300     2.01866 2.00035

# x0 = 10.0
# epoch = 30
# learning_rate = 0.001
# 30      9.53366 58.75609

# x0 = 10.0
# epoch = 300
# learning_rate = 10
# OverflowError: (34, 'Result too large')


# x0 = 5.0
# epoch = 30
# learning_rate = 0.1
# 30      2.00371 2.00001

# x0 = 5.0
# epoch = 30
# learning_rate = 0.01
# 30      3.63645 4.67798

# x0 = 5.0
# epoch = 30
# learning_rate = 0.001
# 30      4.82512 9.98132

# x0 = 5.0
# epoch = 30
# learning_rate = 10
# 30      691399853691585612872790897737273966592.00000   478033757684745943109608411413426907044999145500187510551414351763533511786496.00000