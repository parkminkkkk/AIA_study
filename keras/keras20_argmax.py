import numpy as np 
a = np.array([[1,2,3], [6,4,5], [7,9,2], [3,2,1], [2,3,1]])
print(a) 
'''
[[1 2 3]
 [6 4 5]
 [7 9 2]
 [3 2 1]
 [2 3 1]]
 '''
 
print(a.shape)  #(5,3)
print(np.argmax(a)) #7 (가장 높은 수(9)의 자리/위치를 반환) 
print(np.argmax(a, axis=0)) 
#[2 2 1] axis=0 행을 기준으로 (행끼리,세로로) 비교해서 가장 큰 수의 위치를 반환함
#axis=0 행 : 행끼리 비교
print(np.argmax(a, axis=1)) 
# [2 0 1 0 1] axis=1 열을 기준으로 (열끼리, 가로로) 비교해서 가장 큰 수의 위치를 반환
#axis=1 열 : 열끼리 비교
print(np.argmax(a, axis=-1))
#[2 0 1 0 1] 
#axis=-1 : 가장 마지막 (가장 마지막 축)
#이건 2차원이니까 가장 마지막 축은 1 /따라서, -1을 쓰면 이 데이터의 경우 1과 동일 


#axis=1, axis=-1 동일 
#왜냐하면 대체로 one-hot을 사용하는데 one-hot encoding하면 2차원으로 데이터가 나오니까 동일함!

