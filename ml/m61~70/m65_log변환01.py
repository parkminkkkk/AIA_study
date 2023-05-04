#로그변환 : y값 차이가 너무 많이 날때 log로 변환을 해주면서 큰수를 작은수로 줄여주면서 값의 차이범위는 가져감 

import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size = 1000)

#로그변환 
log_data = np.log(data)

#원본 데이터 히스토그램 그리기
plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='blue', alpha=0.5)
plt.title('Log Transfored Data')

plt.show()

#한쪽으로 치우친 데이터 -> log변환하면, 정규분포 형태로 나옴