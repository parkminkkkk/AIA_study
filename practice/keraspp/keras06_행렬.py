import numpy as np


x1 = np.array([[1,2], [3,4], [5,6]])
x2 = np.array([[[1,2,3], [4,5,6], [7,8,9]]])
x3 = np.array([[[1]], [[2]], [[3]], [[4]]])
x4 = np.array([[[1,2], [3,4], [5,6]], [[7,8], [9,10], [11,12]]])
x5 = np.array([[[1,2,3]], [[4,5,6]]])
x6 = np.array([[1], [2], [3], [4], [5]])
x7 = np.array([[[1], [2]], [[3], [4]], [[5], [6]]])
x8 = np.array([[[1], [2], [3]]])
x9 = np.array([[[1,2]], [[3,4]], [[5,6]]])
x10 = np.array([[[[1,2,3],[4,5,6]]]])

print(x1.shape) #(3, 2)
print(x2.shape) #(1, 3, 3)
print(x3.shape) #(4, 1, 1)
print(x4.shape) #(2, 3, 2)
print(x5.shape) #(2, 1, 3)
print(x6.shape) #(5, 1)
print(x7.shape) #(3, 2, 1)
print(x8.shape) #(1, 3, 1)
print(x9.shape) #(3, 1, 2)
print(x10.shape) # (1, 1, 2, 3)