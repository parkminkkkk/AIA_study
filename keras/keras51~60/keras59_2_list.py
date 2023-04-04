import numpy as np
import pandas as pd

a= [[1,2,3],[4,5,6]]

b= np.array(a)
print(b)
'''
[[1 2 3]
 [4 5 6]]
'''

c = [[1,2,3],[4,5]] 
print(c) 
#[[1, 2, 3], [4, 5]] : list에 들어가는 것은 크기가 달라도 상관 없다. 
d= np.array(c)
print(d) #[list([1, 2, 3]) list([4, 5])] 

#1. np는 규격이 딱 맞아야함/ list는 크기가 달라도 상관이 없다
######################################################################

e = [[1,2,3], ["바보", "맹구", 5,6]]
print(e)  
# [[1, 2, 3], ['바보', '맹구', 5, 6]]
f = np.array(e)
print(f) # [list([1, 2, 3]) list(['바보', '맹구', 5, 6])] : np는 규격 안에 넣어만 줌 , 그러나 연산은 안됨 

#2. list안에는 각각 다른 자료형을 넣어도 상관없다 / np는 안됨
######################################################################
# print(e.shape) #AttributeError: 'list' object has no attribute 'shape'
#list는 크기가 다른 다차원 만들 수 있음.(1,3)/(1,4)두개 같이 넣을 수 있음 -> 따라서 shape 제공하지 않음 / len으로 확인
print(len(e)) #2
print(len(e[0])) #3
print(len(e[1])) #4

#pandas는 수치만 됨 
