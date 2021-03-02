import numpy as np
import matplotlib.pyplot as plt

def selu(x, alp):
    return (x > 0) * x + (x <= 0) * (alp * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)    
y = selu(x, alp=2)

plt.plot(x, y)
plt.grid()
plt.show()