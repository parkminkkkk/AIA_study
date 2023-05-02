import numpy as np 
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Ridge, Lasso #딥러닝 레이어층에서 가중치 규제 - L1 절대값(라쏘) 규제, L2 제곱(릿지) 규제
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
x, y = load_linnerud(return_X_y=True)
# print(x)
# print(y)
print(x.shape)  #(20, 3)
print(y.shape)  #(20, 3) y가 3개짜리 데이터 


# # 원래 [[2,110,43]] => 예상 y값 :  [[138.  33.  68.]]
# model = Ridge()
# model.fit(x, y)
# y_predict = model.predict(x)
# # print("score:", model.score(x, y))  # score: 0.29687777631731227
# print(model.__class__.__name__, "mae:",
#       round(mean_absolute_error(y, y_predict),4))
# print(model.predict([[2,110,43]]))  #  [[187.32842123  37.0873515   55.40215097]]   
####################################################################################

# # 원래 [[2,110,43]] => 예상 y값 :  [[138.  33.  68.]]
# model = XGBRegressor()
# model.fit(x, y)
# y_predict = model.predict(x)
# # print("score:", model.score(x, y))  # score: score: 0.9999999567184008
# print(model.__class__.__name__, "mae:",
#       round(mean_absolute_error(y, y_predict),4))
# print(model.predict([[2,110,43]]))  # [[138.00215   33.001656  67.99831 ]]  

####################################################################################

# 원래 [[2,110,43]] => 예상 y값 :  [[138.  33.  68.]]
# model = LGBMRegressor()
# model.fit(x, y)
# print("score:", model.score(x, y))  
# print(model.predict([[2,110,43]]))  

###LGBM 에러###
# ValueError: y should be a 1d array, got an array of shape (20, 3) instead.
# LGBM : 다차원y값은 사용xx => 이럴 경우, 훈련 3번 해야함!! 
# 1. ([0],[1],[2])각 각 훈련하고, 결과를 concat해야함
# 2. MultiOutputRegressor
#---------------------------------------------------------------------------------#
# model = MultiOutputRegressor(LGBMRegressor())
# model.fit(x, y)

# y_predict = model.predict(x)
# # print("score:", model.score(x, y))   #score: 0.0
# print(model.__class__.__name__, "mae:",
#       round(mean_absolute_error(y, y_predict),4))   #mae: 8.909999999999998
# print(model.predict([[2,110,43]]))                #[[178.6  35.4  56.1]]


'''
#Ridge
mae: 7.4569
[[187.32842123  37.0873515   55.40215097]]

#xgb
mae: 0.0008
[[138.00215   33.001656  67.99831 ]]

#Multi_lgbm 
mae: 8.91
[[178.6  35.4  56.1]]
'''
####################################################################################
# model = CatBoostRegressor()
# model.fit(x, y)

# y_predict = model.predict(x)
# # print("score:", model.score(x, y))   #score: 0.0
# print(model.__class__.__name__, "mae:",
#       round(mean_absolute_error(y, y_predict),4))   #mae: 8.909999999999998
# print(model.predict([[2,110,43]]))                #[[178.6  35.4  56.1]]

###Catboost 에러###
#Currently only multi-regression, multilabel and survival objectives work with multidimensional target
#---------------------------------------------------------------------------------#

# 방법1. 
# model = MultiOutputRegressor(CatBoostRegressor(verbose=0))
# model.fit(x, y)

# y_predict = model.predict(x)
# # print("score:", model.score(x, y))   #score: 0.0
# print(model.__class__.__name__, "mae:",
#       round(mean_absolute_error(y, y_predict),4))   #mae: 0.2154
# print(model.predict([[2,110,43]]))                  #[[138.97756017  33.09066774  67.61547996]]

# 방법2. 
model = CatBoostRegressor(loss_function='MultiRMSE')  #다형으로 계산하는 경우에는 가능!
model.fit(x, y)  

y_predict = model.predict(x)
# print("score:", model.score(x, y))   #score: 0.0
print(model.__class__.__name__, "mae:",
      round(mean_absolute_error(y, y_predict),4))   #mae: 0.0638
print(model.predict([[2,110,43]]))                  #[[138.21649371  32.99740595  67.8741709 ]]
