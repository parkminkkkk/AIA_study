#BayesianOptimization => 최댓값 찾아주는 알고리즘 
#분류모델 - ACC의 최댓값  
#회귀모델 - R2의 최댓값 보면 좋지만 모델 성능이 안좋을 경우 R2신빙성 낮음 / RMSE, MAE에 마이너스 붙여줌(return에서 -1.0곱해줌) : 작을 수록 좋기때문에 


#1. 파라미터 범위지정
param_bounds = {'x1' : (-1, 5),     # 키(텍스트) : 밸류(튜플형테)로 넣어줘야함!!
                'x2' : (0, 4)}

#2. 함수정의
def y_function(x1, x2):
    return -x1**2 - (x2 -2)**2 +10


#3. 최댓값 찾기
# pip install bayesian-optimization
from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(
    f = y_function,
    pbounds = param_bounds,
    random_state = 337
)

optimizer.maximize(init_points=2,   #두개의 랜덤값
                   n_iter=20,       #20번 파라미터 검색 횟수(훈련)        => 총 22번 
                   )
print(optimizer.max)

