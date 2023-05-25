import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


y= np.array([[1,2,3,4,5,6,7,8,9,10]]) 
print(y.shape) # (1,10)
#y = y.T # (10.1) 

y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
print(y.shape) # (1, 10)
y = y.T  
print(y.shape) # (10, 1)

#mlp3.py
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print(y.shape) # (10, )
y = y.T # (10.) 

#행과 열 중 하나가 0일 경우, transpose해도 동일하게 됨
