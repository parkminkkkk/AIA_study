import numpy as np

datasets = np.array(range(1,41)).reshape(10,4) #벡터형태, 1차원->reshape,2차원  
print(datasets)                                #(10,4) 2차원
'''
[[ 1  2  3  4] 
 [ 5  6  7  8] 
 [ 9 10 11 12] 
 [13 14 15 16] 
 [17 18 19 20] 
 [21 22 23 24] 
 [25 26 27 28] 
 [29 30 31 32] 
 [33 34 35 36] 
 [37 38 39 40]]
'''
print(datasets.shape) #(10,4)

# x_data = datasets[:, :3]  
x_data = datasets[:, :-1]  
y_data = datasets[:, -1]  

print(x_data) 
print(y_data) 
print(x_data.shape, y_data.shape) #(10, 3) (10,)


#(10분뒤의 데이터를 맞추는 소스)
#이제 x를 시계열 데이터로 자를 것임 

timesteps = 5

########## x만들기 ###########
def split_x(dataset, timesteps):                   
    aaa = []                                       #aaa라는 빈 공간list만들어 놓음 
    for i in range(len(dataset) - timesteps):      #(length : 10) - 5 + 1 = 6  # 즉, for i in 6 : 6번 반복하겠다(0.1.2.3.4.5) i=번마다 한칸씩 올라감 
        subset = dataset[i : (i + timesteps)]      #[0~5] 라는 데이터셋이 subset데이터값에 0,1,2,3,4,5개 들어감 
        aaa.append(subset)                         #append : aaa의 list에 넣어라     
    return np.array(aaa)                           # i 에 012345개 차례대로 들어가면서 반복됨 

x_data = split_x(x_data, timesteps)
print(x_data) 
'''
[[[ 1  2  3]
  [ 5  6  7]
  [ 9 10 11]
  [13 14 15]
  [17 18 19]]

 [[ 5  6  7]
  [ 9 10 11]
  [13 14 15]
  [17 18 19]
  [21 22 23]]

 [[ 9 10 11]
  [13 14 15]
  [17 18 19]
  [21 22 23]
  [25 26 27]]

 [[13 14 15]
  [17 18 19]
  [21 22 23]
  [25 26 27]
  [29 30 31]]

 [[17 18 19]
  [21 22 23]
  [25 26 27]
  [29 30 31]
  [33 34 35]]

 [[21 22 23]
  [25 26 27]
  [29 30 31]
  [33 34 35]
  [37 38 39]]]
(6, 5, 3)
'''
print(x_data.shape) #(6, 5, 3)

#마지막 데이터는 사용하지 못하니까 미리 뺴주기 (for문)
#for i in range(len(dataset) - timesteps): # (5,5,3)

#########y 만들기############
#timesteps의 개수만큼 뺴기 
y_data = y_data[timesteps:]
print(y_data)


