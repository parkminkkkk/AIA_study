# 난 정말 시그모이드 ~ ~ ♪ ♬
#한정 함수 : 범위내 수로 한정해서 다음 layer로 넘겨주는 것 
# 0~1 
 
import numpy as np
import matplotlib.pyplot as plt 

# def sigmoid(x):
#     return 1/ (1+ np.exp(-x))

sigmoid = lambda x: 1/ (1+ np.exp(-x))

x = np.arange(-5, 5, 0.1)
print(x)
print(len(x))  #100

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()

