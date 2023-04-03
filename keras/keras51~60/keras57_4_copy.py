import numpy as np

aaa = np.array([1,2,3])
bbb = aaa
print(bbb) #[1 2 3]

bbb[0] = 4
print(bbb) #[4 2 3]
print(aaa) #[4 2 3] 
#aaa 건들지 않아도 변경됨 
#aaa의 주소값이 bbb로 들어감(bbb는 aaa의 주소값을 참고해서 만듦)
#numpy의 주소값이 공유됨(메모리가 공유되므로 같이 변경됨) => 문제 발생 가능**

print("===============================")
ccc = aaa.copy()  #새로운 메모리 구조 생성됨 (문제해결)
ccc[1] = 7

print(ccc) #[4 7 3]
print(aaa) #[4 2 3]
