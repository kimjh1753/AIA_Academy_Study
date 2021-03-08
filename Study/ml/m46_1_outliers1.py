# 이상치 처리
# 1. 0 처리
# 2. Nan 처리 후 보관
# 3.4.5... 알아서 해

import numpy as np
# aaa = np.array([1,2,3,4,6,7,90,100,5000,10000])
aaa1 = np.array([1,2,3,4,6,5000,10000,7,90,100])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50 ,75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

# outlier_loc = outliers(aaa)
outlier_loc = outliers(aaa1)
print("이상치의 위치 : ", outlier_loc) 

# 실습
# 위 aaa 데이터를 boxplot으로 그리시오!!

import matplotlib.pyplot as plt

plt.boxplot(aaa1)
plt.show()