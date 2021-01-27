import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
