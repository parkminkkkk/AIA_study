import numpy as np

dataset = np.array(range(1, 11)) #1~10
timesteps = 5 #5개씩 자르기 

def split_x(dataset, timesteps):                   
    aaa = []                                       #aaa라는 빈 공간list만들어 놓음 
    for i in range(len(dataset) - timesteps +1):   #(length : 10) - 5 + 1 = 6  # 즉, for i in 6 : 6번 반복하겠다(0.1.2.3.4.5) i=번마다 한칸씩 올라감 
        subset = dataset[i : (i + timesteps)]      #[0~5] 라는 데이터셋이 subset데이터값에 0,1,2,3,4,5개 들어감 
        aaa.append(subset)                         #append : aaa의 list에 넣어라     
    return np.array(aaa)                           # i 에 012345개 차례대로 들어가면서 반복됨 

bbb = split_x(dataset, timesteps)
print(bbb)           
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]   #6번 반복, 
'''
print(bbb.shape)     
#(6, 5)


##########데이터 x,y####################
x = bbb[:, :4] # x = bbb[:, :-1]
y = bbb[:, -1:]
print(x) 
'''
[[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]
'''
print(y) 
'''
[[ 5]
 [ 6]
 [ 7]
 [ 8]
 [ 9]
 [10]]
'''

 #[참고]https://www.w3schools.com/python/numpy/numpy_array_slicing.asp

