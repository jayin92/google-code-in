import numpy as np
import matplotlib.pyplot as plt
from time import time

def normal_mul(a, b):
    size = a.shape[0]
    res = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            for k in range(size):
                res[i][j] += a[i][k] * b[k][j]
    
    return res

if __name__ == "__main__":
    x = []
    normal_y = []
    np_y = []
    for n in range(1, 10 ** 2):
        size = n
        x.append(size)

        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        start = time()
        normal_mul(a, b)
        normal_time = time() - start
        print(normal_time)
        normal_y.append(normal_time)
        start = time()
        np.matmul(a, b)
        np_time = time() - start
        print(np_time)
        np_y.append(np_time)

    plt.plot(x, normal_y)
    plt.plot(x, np_y)

    plt.legend(['Normal Python', 'Numpy matmul'], loc="upper left")
    plt.show()

