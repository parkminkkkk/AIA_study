#데이터프레임형식의 이상치 확인 

import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
aaa= np.transpose(aaa) #(13,2)
print(aaa)

#이상치 찾는 함수(df)
def outilers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75], axis=0)
    print("1사분위 : ", quartile_1) 
    print("q2 :", q2)               
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)  
    upper_bound = quartile_3 + (iqr * 1.5)  
    outliers = np.where((data_out > upper_bound) | (data_out < lower_bound))
    return list(zip(outliers[0], outliers[1])) #outliers[0],outliers[1]-> 1차원 배열

outilers_loc = outilers(aaa)
print("이상치의 위치 : ", outilers_loc)

#df_각 컬런에 대한 이상치 값과 위치 찾는 함수
# def outliers(data_out):
#     for i in range(data_out.shape[1]):
#         quartile_1, q2, quartile_3 = np.percentile(data_out[:,i], [25, 50, 75])
#         # print("1사분위 :", quartile_1)
#         # print("q2 : ", q2)            
#         # print("3사분위 :", quartile_3)
#         iqr = quartile_3 - quartile_1
#         # print("iqr :", iqr)
#         lower_bound = quartile_1 - (iqr * 1.5)
#         upper_bound = quartile_3 + (iqr * 1.5)
#         # print('최솟값:', lower_bound)                                            
#         # print('최댓값:', upper_bound)
#         outlier_indices = np.where((data_out[:,i] > upper_bound) | (data_out[:,i] < lower_bound))[0]
#         # print(outlier_indices)
#         # whereoi = np.where((data_out[:,i] > upper_bound) | (data_out[:,i] < lower_bound))
#         if outlier_indices.size > 0:
#             print(i+1,"번째 컬런의 이상치 :", data_out[outlier_indices,i],' 이상치의 위치 :', outlier_indices)
#         else:
#             print(i+1,"번째 컬런 이상치 없음")
# outliers(aaa)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

'''
1사분위 : 4.0
q2 :  7.0
3사분위 : 10.0
iqr : 6.0
최솟값: -5.0
최댓값: 19.0
1 번째 컬런의 이상치 : [-10  50]  이상치의 위치 : [ 0 12]

1사분위 : 200.0
q2 :  400.0
3사분위 : 600.0
iqr : 400.0
최솟값: -400.0
최댓값: 1200.0
2 번째 컬런의 이상치 : [-70000]  이상치의 위치 : [6]
'''

'''
##이상치 찾는 함수 => df형태는 오류 발생함##
def outliers(data_out): 
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75]    #np.percentile : 백분위수
                                               )
    print("1사분위 :", quartile_1)
    print("q2 : ", q2)              #중위값
    print("3사분위 :", quartile_3)   # 0.75* 13(데이터의 n) = 9.75(반올림=10) 10번째자리의 값 =10
    iqr = quartile_3-quartile_1     #중간 50%의 확산을 측정
    print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)     #하한은 1사분위수에서 IQR의 1.5배를 빼서 계산, 허용 가능한 값 범위의 하한을 정의
    upper_bound = quartile_3 + (iqr * 1.5)      # 상한은 IQR의 1.5배를 세 번째 사분위수에 더하여 계산, 허용 가능한 값 범위의 상한을 정의
    print(lower_bound)                                            
    print(upper_bound)
    return np.where((data_out>upper_bound) |   #허용 가능한 값 범위를 벗어나는 데이터 세트의 잠재적 이상값을 식별
                     (data_out<lower_bound))   # | : or  (upper_bound보다 크거나, lower_bound보다 작은 것 반환)
                                                #np.where : 인덱스 배열을 반환(array), 위치반환 
outliers_loc = outliers(aaa)
print("이상치의 위치 :", outliers_loc)
'''



