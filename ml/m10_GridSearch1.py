#GridSearch : 파라미터 전체를 다 조사해보겠다. (hyperparameter + cross_validation)
#파라미터 : model, model.fit 두가지에 파라미터 사용함 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

#1. 데이터 
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, test_size=0.2, stratify=y
)


gamma = [0.001, 0.01, 0.1, 1 , 10, 100]
C = [0.001, 0.01, 0.1, 1, 10, 100]

max_score = 0

for i in gamma:
    for j in C:
        #2. 모델
        model = SVC(gamma=i, C=j)

        #3. 컴파일, 훈련
        model.fit(x_train, y_train)

        #4. 평가, 예측 
        score = model.score(x_test, y_test)
        # print("acc:", score) # acc: 0.9666666666666667

        if max_score < score:
            max_score = score
            best_parameters = {'gamma': i , 'C': j} #if문 안에 있으므로 max_score와 best_parameters는 항상 같이 간다(최고점수일때만 갱신되므로)

print("최고점수:", max_score)
print("최적의 매개변수:", best_parameters)

# 최고점수: 1.0
# 최적의 매개변수 : {'gamma': 0.01, 'C': 100}