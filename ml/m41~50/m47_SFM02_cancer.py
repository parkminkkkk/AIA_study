import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_covtype, load_digits
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score
import warnings
warnings.filterwarnings(action='ignore')
import sklearn as sk 
print(sk.__version__) #1.2.2


#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, #stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' : 10000,
              'learning_rate' : 0.01,
              'max_depth': 3,
              'gamma': 0,
              'min_child_weight': 0,
              'subsample': 0.4,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.7,
              'colsample_bynode': 0,
              'reg_alpha': 1,
              'reg_lambda': 1,
              'random_state' : 123,
            #   'eval_metric' : 'error'
              }

#2. 모델
model = XGBClassifier(**parameters)
model.set_params(early_stopping_rounds =100, eval_metric = 'error', **parameters)    


# train the model
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose =0
          )

#4. 평가
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
print(f"accuracy_score: {acc}")
print(f"RMSE: {rmse}")
print("======================================")

##################################################
print("컬럼 중요도:",model.feature_importances_)
thresholds = np.sort(model.feature_importances_)        #list형태로 들어가있음. #np.sort오름차순 정렬
print("컬럼 중요도(오름차순):", thresholds)
print("======================================")

from sklearn.feature_selection import SelectFromModel 
for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)    # prefit = False면 다시 훈련 / True :사전훈련 

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print("변형된 x_train:", select_x_train.shape, "변형된 x_test:", select_x_test.shape)

    selection_model = XGBClassifier()

    selection_model.set_params(early_stopping_rounds =10, eval_metric = 'error', **parameters)

    selection_model.fit(select_x_train, y_train,
                        eval_set = [(select_x_train, y_train), (select_x_test, y_test)],
                        verbose =0)
    
    select_y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)

    print("Tresh=%.3f, n=%d, acc: %.2f%%" %(i, select_x_train.shape[1], score*100))
    #%.3f:부동소수점 => 소수 셋쨰짜리까지 / %d : 정수형 / %% : 두개 입력해야 하나의 %로 인식함'%'
    #첫번째 % = i/ 두번째% = select_x_train.shape[1] / R2% = score*100 


# prefit = False면 다시 훈련 / True :사전훈련 
# #threshold : 특정 값 이상만 선택을 하겠다. (list형식 안받음, list개수만큼 넣어줌)
# 처음엔 10개 다 돌아감, 두번쨰엔 9개, 8개, 7개 ,,,  (SelectFromModel 내부에 컬럼을 한개씩 삭제하는 기능 포함되어 있음)

'''
accuracy_score: 0.9473684210526315
RMSE: 0.22941573387056177
======================================
Tresh=0.005, n=30, acc: 92.11%
Tresh=0.005, n=29, acc: 95.61%
Tresh=0.007, n=28, acc: 94.74%
Tresh=0.008, n=27, acc: 94.74%
Tresh=0.009, n=26, acc: 94.74%
Tresh=0.010, n=25, acc: 94.74%
Tresh=0.012, n=24, acc: 93.86%
Tresh=0.015, n=23, acc: 97.37%
Tresh=0.015, n=22, acc: 95.61%
Tresh=0.016, n=21, acc: 93.86%
Tresh=0.020, n=20, acc: 93.86%
Tresh=0.022, n=19, acc: 96.49%
Tresh=0.026, n=18, acc: 94.74%
Tresh=0.026, n=17, acc: 93.86%
Tresh=0.027, n=16, acc: 93.86%
Tresh=0.031, n=15, acc: 93.86%
Tresh=0.034, n=14, acc: 94.74%
Tresh=0.037, n=13, acc: 93.86%
Tresh=0.043, n=12, acc: 93.86%
Tresh=0.045, n=11, acc: 93.86%
Tresh=0.049, n=10, acc: 95.61%
Tresh=0.050, n=9, acc: 95.61% ***
Tresh=0.055, n=8, acc: 93.86%
Tresh=0.057, n=7, acc: 91.23%
Tresh=0.057, n=6, acc: 90.35%
Tresh=0.060, n=5, acc: 88.60%
Tresh=0.061, n=4, acc: 90.35%
Tresh=0.063, n=3, acc: 89.47%
Tresh=0.066, n=2, acc: 89.47%
Tresh=0.068, n=1, acc: 88.60%
'''