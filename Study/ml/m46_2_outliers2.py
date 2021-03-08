# 실습
# outliers1을 행렬형태도 적용할 수 있도록 수정

import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
               [1000, 2000, 3, 4000, 5000, 6000, 7000, 8000, 9000, 10000]])
aaa = aaa.transpose()
print(aaa.shape)    # (10, 2)

def outliers(data_out):
    loc = []
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50 ,75], axis=0, keepdims=True)
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    print(lower_bound)
    upper_bound = quartile_3 + (iqr * 1.5)
    print(upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

# outlier_loc = outliers(aaa)
outlier_loc = outliers(aaa)
print("이상치의 위치 : ", outlier_loc) 

# 실습
# 위 aaa 데이터를 boxplot으로 그리시오!!

import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.show()