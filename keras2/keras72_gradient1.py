import numpy as np
import matplotlib.pyplot as plt 
 
f = lambda x: x**2 -4*x  +6 

x = np.linspace(-1, 6, 100)  #-1부터 6까지 사이의 숫자 100개 

print(x, len(x))     #100 

y = f(x)

plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# gradient가 y = 0이 되는 지점이 x의 최솟값  #f(x)함수 미분한 함수 : 그 함수의 기울기를 찾는 함수 \
# 미분하면 2x-4 = 0 , x = 2