#실습 
#모델 : RandomForestClassifier
# parameters = [
#     {'n_estimators' : [100,200]},
#     {'max_depth' : [6,8,10,12]},
#     {'min_samples_leaf' : [3,5,7,10]},
#     {'min_samples_split' : [2,3,5,10]},
#     {'n_jobs' : [-1, 2, 4]}]
#파라미터 조합으로 2개 이상 엮을 것
####################################################
import time
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV   
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import accuracy_score

#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, test_size=0.2
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)


parameters = [
    {'n_estimators' : [100,200], 'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10], 'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10]},
  ]

#2. 모델 
model = HalvingGridSearchCV(RandomForestClassifier(), parameters, factor=3,
                     cv=kfold, verbose=1, refit=True, n_jobs=-1)

#3. 컴파일, 훈련 
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수:", model.best_estimator_) 
print("최적의 파라미터:", model.best_params_)
print("best_score:", model.best_score_)
print("model.score:", model.score(x_test, y_test))
print("걸린시간 :", round(end_time-start_time,2), "초")

#4. 평가, 예측
y_predict = model.predict(x_test)
print("accuracy_score:", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)            
print("최적 튠 ACC:", accuracy_score(y_test, y_pred_best))

#
'''
최적의 매개변수: RandomForestClassifier(min_samples_split=3)
최적의 파라미터: {'min_samples_split': 3}
best_score: 0.9444444444444444
model.score: 0.9649122807017544
걸린시간 : 22.56 초
accuracy_score: 0.9649122807017544
최적 튠 ACC: 0.9649122807017544
'''
#
'''
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수: RandomForestClassifier(max_depth=8, min_samples_leaf=5, n_estimators=200)
최적의 파라미터: {'n_estimators': 200, 'min_samples_leaf': 5, 'max_depth': 8}
best_score: 0.9604395604395606
model.score: 0.9649122807017544
걸린시간 : 6.85 초
accuracy_score: 0.9649122807017544
최적 튠 ACC: 0.9649122807017544
'''
#
'''
Fitting 5 folds for each of 68 candidates, totalling 340 fits
최적의 매개변수: RandomForestClassifier(max_depth=6, min_samples_leaf=5)     
최적의 파라미터: {'max_depth': 6, 'min_samples_leaf': 5, 'n_estimators': 100}
best_score: 0.9670329670329672
model.score: 0.9649122807017544
걸린시간 : 20.79 초
accuracy_score: 0.9649122807017544
최적 튠 ACC: 0.9649122807017544
'''