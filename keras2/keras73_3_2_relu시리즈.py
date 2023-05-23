# [실습]
# elu, selu, reaky_relu 
# relu시리즈 : 음수값에 대한 소멸이 아깝다=> 음수값을 조금씩 변형을 줌 

import numpy as np 
import matplotlib.pyplot as plt 

#1. relu---------------------------------------------------------------------------------
# relu = lambda x: np.maximum(0, x)   #relu :  0 ~ x로 한정하는 함수 
def relu(x):
    return np.maximum(0, x)   
# 0과 x를 비교해서 큰 값 뽑아냄 => 즉, 음수는 0으로, 양수는 양수값 x그대로 출력 
# ReLU는 단순성과 기울기 소실 문제를 처리할 수 있는 능력

#2. ELU(Exponential linear unit)---------------------------------------------------------
def elu(x, alpha=1.0):     
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))  #alpha음수 값의 출력을 제어하는 ​​하이퍼파라미터
#ELU는 입력에 대해 음수 값을 허용하므로 죽어가는 ReLU 문제를 완화하는 데 도움

#3. selu(scaled elu)----------------------------------------------------------------------
def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
#SELU는 입력에 대해 평균을 0에 가깝게 유지하고 표준 편차를 1에 가깝게 유지하도록 설계


#4. leaky_relu---------------------------------------------------------
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)   #alpha는 작은 양의 상수
# 음수일 때 그래디언트가 0.01이라는 점을 제외하고는 ReLU와 같은 특성을 지님 
# leaky_relu는 음수 입력에 대해 작고 0이 아닌 기울기를 허용하여 ReLU 함수에서 "죽은" 뉴런 문제를 해결
# 음의 입력에 대해 작은 음의 기울기를 도입함으로써 Leaky ReLU는 값이 0인 기울기를 방지

#5. RReLU (Randomized leaky ReLU)---------------------------------------------------------
def rrelu(x, lower=0.01, upper=0.1):
    alpha = np.random.uniform(lower, upper, size=x.shape)
    return np.maximum(alpha * x, x)
#RReLU 함수는 학습 중에 지정된 범위 내에서 음의 기울기가 무작위로 선택되고 추론 중에 고정되는 Leaky ReLU의 변형
#교육 중에 RReLU는 균일한 분포에서 음의 기울기를 샘플링하여 임의성을 도입

#6. PReLU (Parametric leaky ReLU)---------------------------------------------------------
def prelu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)
# PReLU 기능은 Leaky ReLU와 유사하지만 고정 값을 사용하는 대신 교육 중에 음의 기울기를 학습
# PReLU는 각 뉴런에 대한 최적의 기울기를 학습하여 잠재적으로 모델 성능을 향상 가능 

x = np.arange(-5, 5, 0.1)
y_relu = relu(x)
y_elu = elu(x)
y_selu = selu(x)
y_leaky_relu = leaky_relu(x)
y_rrelu = rrelu(x)
y_prelu = prelu(x)


plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_elu, label='ELU')
plt.plot(x, y_selu, label='SELU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.plot(x, y_rrelu, label='RReLU')
plt.plot(x, y_prelu, label='PReLU')
plt.legend()
plt.grid()
plt.show()


