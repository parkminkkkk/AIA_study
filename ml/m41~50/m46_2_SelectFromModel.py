#save_model : 가중치 save

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
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
x, y = load_diabetes(return_X_y=True)


# features =['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

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
model = XGBRegressor(**parameters)
model.set_params(early_stopping_rounds =100, eval_metric = 'rmse', **parameters)    


# train the model
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose =0
          )

#4. 평가
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
print(f"R2 score: {r2}")
print(f"RMSE: {rmse}")
print("======================================")

##################################################
print(model.feature_importances_)
# [0.06504052 0.02900812 0.20614523 0.10602617 0.06462499 0.06759115 0.10111099 0.08934037 0.16957761 0.10153492]
thresholds = np.sort(model.feature_importances_)        #list형태로 들어가있음. #np.sort오름차순 정렬
print(thresholds)
# [0.02900812 0.06462499 0.06504052 0.06759115 0.08934037 0.10111099 0.10153492 0.10602617 0.16957761 0.20614523]

from sklearn.feature_selection import SelectFromModel 
for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)    # prefit = False면 다시 훈련 / True :사전훈련 

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print("변형된 x_train:", select_x_train.shape, "변형된 x_test:", select_x_test.shape)

    selection_model = XGBRegressor()

    selection_model.set_params(early_stopping_rounds =10, eval_metric = 'rmse', **parameters)

    selection_model.fit(select_x_train, y_train,
                        eval_set = [(select_x_train, y_train), (select_x_test, y_test)],
                        verbose =0)
    
    select_y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)

    print("Tresh=%.3f, n=%d, R2: %.2f%%" %(i, select_x_train.shape[1], score*100))
    #%.3f:부동소수점 => 소수 셋쨰짜리까지 / %d : 정수형 / %% : 두개 입력해야 하나의 %로 인식함'%'
    #첫번째 % = i/ 두번째% = select_x_train.shape[1] / R2% = score*100 


# prefit = False면 다시 훈련 / True :사전훈련 
# #threshold : 특정 값 이상만 선택을 하겠다. (list형식 안받음, list개수만큼 넣어줌)
# 처음엔 10개 다 돌아감, 두번쨰엔 9개, 8개, 7개 ,,,  (SelectFromModel 내부에 컬럼을 한개씩 삭제하는 기능 포함되어 있음)

'''
#Tresh : model.feature_importances_의 값 
#n : 컬럼의 수 
------------------------------------------------------
변형된 x_train: (353, 10) 변형된 x_test: (89, 10)
Tresh=0.029, n=10, R2: 47.94%
변형된 x_train: (353, 9) 변형된 x_test: (89, 9)
Tresh=0.065, n=9, R2: 44.46%
변형된 x_train: (353, 8) 변형된 x_test: (89, 8)
Tresh=0.065, n=8, R2: 46.38%
변형된 x_train: (353, 7) 변형된 x_test: (89, 7)
Tresh=0.068, n=7, R2: 46.65%
변형된 x_train: (353, 6) 변형된 x_test: (89, 6)
Tresh=0.089, n=6, R2: 49.06%
변형된 x_train: (353, 5) 변형된 x_test: (89, 5)
Tresh=0.101, n=5, R2: 48.66%
변형된 x_train: (353, 4) 변형된 x_test: (89, 4)
Tresh=0.102, n=4, R2: 46.23%
변형된 x_train: (353, 3) 변형된 x_test: (89, 3)
Tresh=0.106, n=3, R2: 46.33%
변형된 x_train: (353, 2) 변형된 x_test: (89, 2)
Tresh=0.170, n=2, R2: 43.38%
변형된 x_train: (353, 1) 변형된 x_test: (89, 1)
Tresh=0.206, n=1, R2: 27.46%
------------------------------------------------------
Tresh=0.029, n=10, R2: 47.94%
Tresh=0.065, n=9, R2: 44.46%
Tresh=0.065, n=8, R2: 46.38%
Tresh=0.068, n=7, R2: 46.65%
Tresh=0.089, n=6, R2: 49.06%
Tresh=0.101, n=5, R2: 48.66%
Tresh=0.102, n=4, R2: 46.23%
Tresh=0.106, n=3, R2: 46.33%
Tresh=0.170, n=2, R2: 43.38%
Tresh=0.206, n=1, R2: 27.46%
'''