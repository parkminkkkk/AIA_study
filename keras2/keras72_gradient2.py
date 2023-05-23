import numpy as np
import matplotlib.pyplot as plt 
 
f = lambda x: x**2 -4*x  +6 
# def f(x):
#     return x**2 -4* +6
gradient = lambda x : 2*x-4   #f(x)함수 미분한 함수 : 그 함수의 기울기를 찾는 함수


x = np.linspace(-1, 6, 100)
y = f(x)

#gradient descent찾아가는 과정 
x1 = -10.0    #초기값 
epochs = 200
learning_rate = 0.25 
x_values = np.array([x1])
y_values = np.array([f(x1)])


for i in range(epochs):
    x1 = x1 - learning_rate*gradient(x)
    x_values = np.append(x_values, x1)
    y_values = np.append(y_values, f(x1))

    print(i+1, x, f(x))
    # print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1, x, f(x)))

plt.plot(x, y, 'k-', label='f(x)')
plt.plot(x_values, y_values, 'r.', label='Gradient Descent')
plt.plot(2, f(2), 'sk', label='Minimum')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


'''
1 -4.0 38.0
2 -1.0 11.0
3 0.5 4.25
4 1.25 2.5625
5 1.625 2.140625
6 1.8125 2.03515625
7 1.90625 2.0087890625
8 1.953125 2.002197265625
9 1.9765625 2.00054931640625
10 1.98828125 2.0001373291015625
11 1.994140625 2.0000343322753906
12 1.9970703125 2.0000085830688477
13 1.99853515625 2.000002145767212
14 1.999267578125 2.000000536441803
15 1.9996337890625 2.0000001341104507
16 1.99981689453125 2.0000000335276127
17 1.999908447265625 2.000000008381903
18 1.9999542236328125 2.000000002095476
19 1.9999771118164062 2.000000000523869
20 1.9999885559082031 2.0000000001309672
'''