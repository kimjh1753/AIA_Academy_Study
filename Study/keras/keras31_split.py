import numpy as np

a = np.array(range(1, 11))
size = 5

# def split_x(seq, size):
#     aaa = []
#     for i in range(len(seq) - size + 1):
#         subset = seq[i : (i+size)]
#         aaa.append([item for item in subset])
#     print(type(aaa))
#     return np.array(aaa)

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)    

dataset = split_x(a, size)
print("==========================")
print(dataset)    

# size 5
# <class 'list'>
# ==========================
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

# size 6
# <class 'list'>
# ==========================
# [[ 1  2  3  4  5  6]
#  [ 2  3  4  5  6  7]
#  [ 3  4  5  6  7  8]
#  [ 4  5  6  7  8  9]
#  [ 5  6  7  8  9 10]]

# size 7
# <class 'list'>
# ==========================
# [[ 1  2  3  4  5  6  7]
#  [ 2  3  4  5  6  7  8]
#  [ 3  4  5  6  7  8  9]
#  [ 4  5  6  7  8  9 10]]

# size 8
# <class 'list'>
# ==========================
# [[ 1  2  3  4  5  6  7  8]
#  [ 2  3  4  5  6  7  8  9]
#  [ 3  4  5  6  7  8  9 10]]