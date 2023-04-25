# n_jobs = -1
#     tree_method = 'gpu_hist'
#     predictor = 'gpu_predictor'
#     gpu_id = 0

import time
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV   
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


(x_train, y_train), (x_test, y_test) = mnist.load_data()

pca = PCA()
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

parameters = [
{"n_estimators":[100,200,300], "learning_rate": [0.1, 0.3, 0.001, 0.01], "max_depth":[4,5,6]},
{"n_estimators": [90,100,110], "learning_rate": [0.1, 0.001, 0.01], "max_depth":[4,5,6], "colsample_bytree": [0.6, 0.9, 1]},
{"n_estimators": [90,110], "learning_rate": [0.1, 0.001, 0.5], "max_depth":[4,5,6], "colsample_bytree": [0.6, 0.9, 1], 
 "colsample_bylevel": [0.6,0.7,0.9]}]

#2. 모델 
model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0), parameters,
                     cv=2, verbose=1, refit=True, n_jobs=-1)

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


'''
최적의 파라미터: {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1}
best_score: 0.9523666666666666
model.score: 0.1342
걸린시간 : 34004.61 초
accuracy_score: 0.1342
최적 튠 ACC: 0.1342
'''