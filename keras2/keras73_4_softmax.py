import numpy as np 
import matplotlib.pyplot as plt 

def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x))    #0과 1사이의 수들을 n만큼 나눈 수 

# softmax = lambda x: np.exp(x)/ np.sum(np.exp(x))


x = np.arange(1, 5)
y = softmax(x)  

ratio = y   #원형 차트에 표시할 데이터 값입니다. 이 경우 y숫자 값을 포함하는 목록 또는 배열과 같은 객체로 간주
labels = y 

plt.pie(ratio, labels, shadow=True, startangle=90)

plt.show()




