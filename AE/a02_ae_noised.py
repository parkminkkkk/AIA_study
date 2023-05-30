import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터 
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255.
x_test = x_test.reshape(10000, 784).astype('float32') / 255.

x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape) #약 10프로의 확률을 랜덤하게 넣어줌  
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape) #약 10프로의 확률을 랜덤하게 넣어줌 
#np.random.normal : 노이즈는 평균이 0이고 표준 편차가 0.1인 정규 분포에서 생성
#np.random.uniform : 음수값 없도록 노이즈 줌

print(x_train_noised.shape, x_test_noised.shape)  #(60000,784) (10000, 784)

print(np.max(x_train_noised), np.min(x_train_noised)) #1.4981282905693214 -0.5437005089686505
print(np.max(x_test_noised), np.min(x_test_noised))   #1.449460149539148 -0.49511554036498756

