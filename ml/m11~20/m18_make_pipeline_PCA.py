#PCA :주성분분석
#차원(colum) 축소 = 컬럼 개수 축소 
#즉, 컬럼 압축 개념 (삭제xx:성능 더 좋아질 수도 있음)

import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA     


#1. 데이터 
x,y = load_digits(return_X_y=True)
print(x.shape) #(1797, 64)              #sklearn은 2차원데이터만 받음 (1797,8,8)-> (1797,64)로 받음
print(np.unique(y, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

# #차원 축소(컬러 압축)
# pca = PCA(n_components=8) #n개로 컬럼을 압축할 것인지
# pca = PCA()               #디폴트 : 0개임, 차원축소는 일어나지 않았지만 변환은 되었음!
# x = pca.fit_transform(x)
# print(x.shape) #(1797, 8)
# print(x.shape) #(1797, 64)  # pca = PCA()일때


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
#2. 모델 
# model = RandomForestClassifier()

#scaler +모델 합침
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
# model = make_pipeline(StandardScaler(), RandomForestClassifier())
model = make_pipeline(PCA(n_components=8), StandardScaler(), SVC())


#3. 훈련 
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score:", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score:", acc)

'''
#model = make_pipeline(StandardScaler(), SVC())
model.score: 0.9333333333333333
accuracy_score: 0.9333333333333333
'''

'''
#model = make_pipeline(PCA(), StandardScaler(), SVC())
model.score: 0.975
accuracy_score: 0.975
'''

'''
# model = make_pipeline(PCA(n_components=8), StandardScaler(), SVC())
model.score: 0.9472222222222222
accuracy_score: 0.9472222222222222
'''
