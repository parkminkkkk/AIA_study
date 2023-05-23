import numpy as np 
import matplotlib.pyplot as plt 

def relu(x):
    return np.maximum(0, x)   # 0과 x를 비교해서 큰 값 뽑아냄 => 즉, 음수는 0으로, 양수는 양수값 x그대로 출력 

# relu = lambda x: np.maximum(0, x)   #relu :  0 ~ x로 한정하는 함수 


x = np.arange(-5, 5, 0.1)
y = relu(x)  

plt.plot(x, y)
plt.grid()
plt.show()




