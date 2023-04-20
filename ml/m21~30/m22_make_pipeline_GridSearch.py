import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
#make_pipeline : 함수
#Pipeline : 클래스

#1. 데이터 
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337
)

parameters = [
    {'randomforestclassifier__n_estimators' : [100,200], 'randomforestclassifier__max_depth' : [6,8,10,12], 'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__max_depth' : [6,8,10,12], 'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__min_samples_leaf' : [3,5,7,10], 'randomforestclassifier__min_samples_split' : [2,3,5,10]},
    {'randomforestclassifier__min_samples_split' : [2,3,5,10]}]
#ValueError: Invalid parameter max_depth for estimator Pipeline(steps=[('std', StandardScaler()), ('rf', RandomForestClassifier())]).
#  Check the list of available parameters with `estimator.get_params().keys()`.
#랜덤포레스트의 파라미터를 써야하는데, pipe의 파라미터를 쓰겠다고 명시했으므로 error뜸 
#따라서, pipe의 파라미터로 변경해주면 해결됨!! 
#1.Pipeline = 'rf__'=> 명시해놓은 [변수명__]을 추가 :(변수명+언더바두개__)
#2.make_pipeline = 'randomforestclassifier__' => 변수명 지정안하므로 모델명 전체를 명시해줘야함

#2. 모델구성
# pipe = Pipeline([('std', StandardScaler()), ('rf', RandomForestClassifier())])   
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)


#3. 훈련 
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score:", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score:", acc)

'''
#model = Pipeline([('std', StandardScaler()), ('svc', SVC())])
model.score: 0.9333333333333333
accuracy_score: 0.9333333333333333
'''

'''
model.score: 0.9666666666666667
accuracy_score: 0.9666666666666667
'''