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

#1. 데이터 
datasets = load_diabetes()
x = datasets['data']
y = datasets.target

print(datasets.feature_names)
features = datasets['feature_names']
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

# create a for loop to delete one feature at a time
for i in features:
    # get the index of the current feature
    feature_index = list(features).index(i)
    # delete the current feature from the data
    x_train_temp = np.delete(x_train, feature_index, axis=1)
    x_test_temp = np.delete(x_test, feature_index, axis=1)

    # initialize the XGBRegressor model
    model = XGBRegressor(**parameters)

    # train the model
    model.fit(x_train_temp, y_train,
              eval_set = [(x_train_temp, y_train), (x_test_temp, y_test)],
              early_stopping_rounds = 100,
              eval_metric = 'rmse',
              verbose =0
              )

    # make predictions
    y_predict = model.predict(x_test_temp)

    # evaluate the model
    r2 = r2_score(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    print(f"Deleted feature: {i}")
    print(f"R2 score: {r2}")
    print(f"RMSE: {rmse}")
    print("======================================")


####################################################
print(model.feature_importances_)
'''
최종 점수 : 0.47644650471299665
r2: <function r2_score at 0x000001F7A63A9280>
RMSE: 52.213072910420216
[0.06453435 0.02573609 0.17875978 0.11040898 0.06981922 0.0743337
 0.10291809 0.08199351 0.19161017 0.09988612]
'''

'''
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
Deleted feature: age
R2 score: 0.4742753332152897
RMSE: 52.32122446552758
======================================
Deleted feature: sex
R2 score: 0.44455571296373253
RMSE: 53.77977415208549
======================================
Deleted feature: bmi
R2 score: 0.4256135950597676
RMSE: 54.68910272163883
======================================
Deleted feature: bp
R2 score: 0.4197468762535782
RMSE: 54.9676873746241
======================================
Deleted feature: s1
R2 score: 0.5016740057532556
RMSE: 50.9395956068734
======================================
Deleted feature: s2
R2 score: 0.48784294186294064
RMSE: 51.64167296272178
======================================
Deleted feature: s3
R2 score: 0.46045113429455053
RMSE: 53.0046674233541
======================================
Deleted feature: s4
R2 score: 0.46409482344553643
RMSE: 52.825388292257195
======================================
Deleted feature: s5
R2 score: 0.4126723523675868
RMSE: 55.30175890563016
======================================
Deleted feature: s6
R2 score: 0.4830113575115985
RMSE: 51.88468962660811
======================================
'''
