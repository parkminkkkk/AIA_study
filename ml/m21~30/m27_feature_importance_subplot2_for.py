#feature_importances 
#트리계열에서만 제공함!! model.feature_importances_
#컬럼의 종류에 따라 훈련결과에 악영향을 주는 불필요한 컬럼 존재함 (노이즈)
#=> 컬럼(열, 특성, feature) 걸러내는 작업 필요함 

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

#1. 데이터 
datasets = load_iris()
# datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337
)

#2. 모델구성
models = [DecisionTreeClassifier(),RandomForestClassifier(),
          GradientBoostingClassifier(),XGBClassifier() ]
modelname = ['DTC', 'RFC', 'GBC', 'XGB']

num= [1,2,3,4]

for i, model in enumerate(models):
    # 모델 훈련
    model.fit(x_train, y_train)
    
    # feature importance 그래프 그리기
    import matplotlib.pyplot as plt
    plt.subplot(2, 2, i+1)
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(type(model).__name__)

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model)

# for i in range(4):
#     globals()['model'+str(i)] = models[i]
#     globals()['model'+str(i)].fit(x_train, y_train)
#     plt.subplot(2, 2, i+1)
#     print(globals()['model'+str(i)].feature_importances_)
#     plot_feature_importances(globals()['model'+str(i)])
#     if i == 3:
#         plt.title('XGBClassifier()')
# plt.show()



# for i, v in enumerate(models):
#     model = v
#     modelname[i] = type(model).__name__
#     model.fit(x_train,y_train)
#     import matplotlib.pyplot as plt
#     def plot_feature_importances(model):
#         n_features = datasets.data.shape[1]
#         plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#         plt.yticks(np.arange(n_features), datasets.feature_names)
#         plt.xlabel('Feature Importances')
#         plt.ylabel('Features')
#         plt.ylim(-1, n_features)
#         plt.title(type(model).__name__)
        
#     for j in num:
#         plt.subplot(2,2,j)
#         plot_feature_importances(model)
# plt.show()