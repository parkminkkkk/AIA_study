from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA

#y값 가져오기 싫을때, '빈칸 명시'하는 법 : 파이썬 기초문법 '_' 언더바 명시 
#무조건 4개를 명시해야하므로 1개면 적어서 땡겨올 수 없음 -> '_'명시해줌
#_가 변수로 먹힘(메모리 할당됨...)
(x_train, __), (x_test, _) = mnist.load_data()
# print(__.shape) #(60000,)
# print(_.shape)  #(10000,)

###이미지 데이터를 쭉 핀다면(dnn사용한다면)=? ###
#=>>(70000, 784) 7만개 데이터, 784컬럼이라고 생각할 수 있음
#초반에 0이 몰려있음 => 압축시켜서 0을 줄여주는 것이 성능이 더 잘 나올 가능성 높음 

#x_train, x_test 합치기 (방법2가지)
x = np.concatenate((x_train,x_test), axis=0)  #(70000, 28, 28)
# x = np.append(x_train, x_test, axis=0)      #(70000, 28, 28)
print(x.shape) 

##########실습#######################
#pca를 통해 0.95 이상인 n_components는 몇개?
#0.95 몇개?
#0.99 몇개?
#0.999 몇개?
#1.0 몇개?
#Hint : np.argmax
#####################################

#reshape (pca는 2차원만 받으므로)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])  #(70000,28*28)
print(x.shape) #(70000, 784)

#데이터x 컬럼 축소
pca = PCA(n_components=784)
x = pca.fit_transform(x)

#설명가능한 변화율
pca_EVR = pca.explained_variance_ratio_  
cumsum = np.cumsum(pca_EVR)  #배열의 누적합
print(cumsum)
print(np.argmax(cumsum >= 0.95) +1) #154
print(np.argmax(cumsum >= 0.99) +1) #331
print(np.argmax(cumsum >= 0.999) +1) #486
print(np.argmax(cumsum >= 1.0) +1) #712나옴 -> 보통0부터 시작하므로 +1해줘야 713개 나옴

d95 = np.argmax(cumsum >= 0.95) +1
d99 = np.argmax(cumsum >= 0.99) +1
d999 = np.argmax(cumsum >= 0.999) +1
d100 = np.argmax(cumsum >= 1.0) +1

print(f"0.95이상의 n_components 개수: {d95}")
print(f"0.99이상의 n_components 개수: {d99}")
print(f"0.999이상의 n_components 개수: {d999}")
print(f"1.0이상의 n_components 개수: {d100}")

'''
0.95이상의 n_components 개수: 154
0.99이상의 n_components 개수: 331
0.999이상의 n_components 개수: 486
1.0이상의 n_components 개수: 713
'''