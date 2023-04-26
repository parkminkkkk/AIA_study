import numpy as np
aaa = np.array([-10, 2,3,4,5,6,7,8,9,10,11,12, 50])

#이상치 찾는 함수
def outliers(data_out): 
    quartile_1, q2, quartile_3 = np.percentile(data_out,    #np.percentile : 백분위수
                                               [25,50,75])
    print("1사분위 :", quartile_1)
    print("q2 : ", q2)              #중위값
    print("3사분위 :", quartile_3)   # 0.75* 13(데이터의 n) = 9.75(반올림=10) 10번째자리의 값 =10
    iqr = quartile_3-quartile_1     #중간 50%의 확산을 측정
    print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)     #하한은 1사분위수에서 IQR의 1.5배를 빼서 계산, 허용 가능한 값 범위의 하한을 정의
    upper_bound = quartile_3 + (iqr * 1.5)     # 상한은 IQR의 1.5배를 세 번째 사분위수에 더하여 계산, 허용 가능한 값 범위의 상한을 정의
    return np.where((data_out>upper_bound) |   #허용 가능한 값 범위를 벗어나는 데이터 세트의 잠재적 이상값을 식별
                     (data_out<lower_bound))   # | : or  (upper_bound보다 크거나, lower_bound보다 작은 것 반환)
                                                #np.where : 인덱스 배열을 반환(array), 위치반환 
outliers_loc = outliers(aaa)
print("이상치의 위치 :", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

'''
1사분위 : 4.0
q2 :  7.0
3사분위 : 10.0
iqr : 6.0
이상치의 위치 : (array([ 0, 12], dtype=int64),)
'''

