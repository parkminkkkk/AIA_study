#feature_importances 
#트리계열에서만 제공함!! model.feature_importances_
#컬럼의 종류에 따라 훈련결과에 악영향을 주는 불필요한 컬럼 존재함 (노이즈)
#=> 컬럼(열, 특성, feature) 걸러내는 작업 필요함 
#===================================================================
#xgb는 중요한 컬럼 순으로 중요도를 뽑을 수 있다. 
#즉, 그래프_중요도까지 지원함!
#하이퍼파라미터 최적의 조건을 찾은 이후 plot_importance 해봐야함
#이후 25%이하의 컬럼 삭제해보고 성능 비교

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
model = XGBClassifier()

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
# print("model.score:", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score:", acc)
print(model, ":", model.feature_importances_)


#그림그리기
import matplotlib.pyplot as plt
# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model)

# plot_feature_importances(model)
# plt.show()


#xgb는 중요한 컬럼 순으로 중요도를 뽑을 수 있다. 
#즉, 그래프_중요도까지 지원함!
from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()

#하이퍼파라미터 최적의 조건을 찾은 이후 plot_importance 해봐야함
#이후 25%이하의 컬럼 삭제해보고 성능 비교
