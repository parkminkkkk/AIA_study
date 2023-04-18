#k-fold (교차검증) - 데이터 전처리과정

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x,y = load_iris(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state=42, test_size=0.2)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42) #n_splits=5 : 100%/5이므로 20%씩 #shuffle : 데이터 섞은 다음 20%씩 
# kfold = KFold() #데이터를 훈련시키는 위치에 따라서 결과값의 차이가 크게 날 수 있음

#2. 모델구성 
model = LinearSVC()

#3,4. 컴파일, 훈련,평가, 예측 
scores = cross_val_score(model, x, y, cv=kfold) #모델, 데이터, 크로스발리데이션을 어떤 것으로 쓸것인지(kfold)
# scores = cross_val_score(model, x, y, cv=5) #위에 Kfold정의하지 않은상태에서도 가능함(단, 성능은 고민해볼 문제)/ 즉,kfold는 cross_val해줄 것들을 정의해주는 것이다.  
print(scores) #[1. 0.96666667 0.93333333 0.96666667 0.96666667] #k-fold의 개수만큼 훈련을 시킨다 (n_splits=5)

print('acc:', scores, 
      '\n cross_val_score 평균:', round(np.mean(scores),4))
#acc: [1.         0.96666667 0.93333333 0.96666667 0.96666667]
#cross_val_score 평균: 0.9667


##train_test_split / cross_cal
#train, test split : 데이터 손실이 있지만, 과적합 되지 않도록 하겠다.
#과적합 포기하고 전체 데이터 쓰겠다 : train,test 모두 cross_val 
