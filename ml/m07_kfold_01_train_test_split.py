import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#1. 데이터 
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

#2. 모델구성
model = SVC()

#3.4 컴파일, 훈련, 평가, 예측 
score = cross_val_score(model, x_train, y_train, cv=kfold)    #model.compile, model.fit, model.evaluate 모두 포함되어있음 
print('cross_val_score(ACC):', score, 
      '\nc_val_score(mean_ACC)', round(np.mean(score), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('c_val_predict acc :', accuracy_score(y_test, y_predict))

# cross_val_score(ACC): [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333] 
# c_val_score(mean_ACC) 0.9583
# c_val_predict acc : 0.8333333333333334   ###test_data로 예측한 값이므로 과적합한 만큼 결과가 안 좋게 나올 수도 있음###