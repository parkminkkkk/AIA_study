import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs #샘플데이터 만들 수 있도록 제공
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler

#가우시안 정규분포를 기준으로 샘플 생성 
x, y = make_blobs(random_state=337,
                  n_samples=50,
                  centers=2,     #중심점을 2로 잡겠다 ( 중심 클러스터 개수 = y의 라벨)
                  cluster_std=1  #cluster의 표준편차
                  ) 


print(x)
print(y)  
print(x.shape, y.shape)  #(50, 2) (50,)

###그래프 그리기###
# plt.scatter(x[:, 0], x[:, 1],   # 가로축 #세로축
#             c =y,               # clustering = y
#             edgecolors='black', # 선컬러 '검정'
#             )  
# plt.show()

##========================================================================================##

###subplot그리기###
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,4))


# ax1.scatter(x[:, 0], x[:, 1],   # 가로축 #세로축
#             c =y,               # clustering = y
#             edgecolors='black', # 선컬러 '검정'
#             )  
# ax1.set_title("original")


# scaler = QuantileTransformer(n_quantiles=50) 
# x_trans =scaler.fit_transform(x)

# ax2.scatter(x_trans[:, 0], x_trans[:, 1],   # 가로축 #세로축
#             c =y,               # clustering = y
#             edgecolors='black', # 선컬러 '검정'
#             ) 
# ax2.set_title(type(scaler).__name__)

# plt.show()

##========================================================================================##
##subplot 4개그리기###

plt.rcParams['font.family'] = 'Malgun Gothic'     #한글폰트
fig, ax = plt.subplots(2, 2, figsize=(12,8))

ax[0,0].scatter(x[:, 0], x[:, 1],   # 가로축 #세로축
            c =y,               # clustering = y
            edgecolors='black', # 선컬러 '검정'
            )  
ax[0,0].set_title("오리지널")


scaler = QuantileTransformer(n_quantiles=50) 
x_trans =scaler.fit_transform(x)
ax[0,1].scatter(x_trans[:, 0], x_trans[:, 1],   # 가로축 #세로축
            c =y,               # clustering = y
            edgecolors='black', # 선컬러 '검정'
            )  
ax[0,1].set_title(type(scaler).__name__)


scaler = PowerTransformer() 
x_trans =scaler.fit_transform(x)
ax[1,0].scatter(x[:, 0], x[:, 1],   # 가로축 #세로축
            c =y,               # clustering = y
            edgecolors='black', # 선컬러 '검정'
            )  
ax[1,0].set_title(type(scaler).__name__)


scaler = StandardScaler() 
x_trans =scaler.fit_transform(x)
ax[1,1].scatter(x_trans[:, 0], x_trans[:, 1],   # 가로축 #세로축
            c =y,               # clustering = y
            edgecolors='black', # 선컬러 '검정'
            )  
ax[1,1].set_title(type(scaler).__name__)

plt.show()
