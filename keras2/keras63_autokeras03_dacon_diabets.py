import autokeras as ak
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time


#1. 데이터 
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, test_size=0.2
)


#2. 모델 
model = ak.StructuredDataClassifier(
    overwrite = False,  #True일 경우 모델탐색을 처음부터 다시 함(속도 느림) -> 성능이 너무 안좋을때 True사용하기 / 보통이상 성능이면 True일때 더 성능향상이 됨  
    max_trials=1,       #디폴트 False
)


#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, epochs = 5, validation_split = 0.15)
end = time.time()

###. 최적의 모델 출력 
best_model = model.export_model()
print(best_model.summary())

###. 최적의 모델 저장
# path = './_save/autokeras/'
# best_model.save(path + "keras62_autokeras1.h5")


#4. 평가, 예측 
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('model 결과 :', results)

#               *loss                 *acc
# model 결과 : [0.9791734218597412, 0.6000000238418579]